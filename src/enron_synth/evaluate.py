"""Evaluation harness. Four questions, each answered by independent measurements:

  1. PRIVACY   Did identifying information survive?
               - verifier residual (pipeline's own per-email hard terms)
               - independent global check: 'enron', Houston, unit acronyms, Lotus-Notes '/HOU/ECT', @enron.com,
                 and full names of ANY Enron person seen in the corpus sample (not only the ones in this e-mail)
               - verbatim copying: word 8-gram overlap with the source e-mail
  2. FIDELITY  Is the structure / distribution of real mail preserved?  (length, recipients, greeting, sign-off,
               quoted block, line count; paired ratios + KS test on the length distribution)
  3. REALISM   Does it look like real company mail?  LLM judge (different model family from the generator):
               absolute 1-5 score + blinded pairwise "which is real?" (50 % = indistinguishable)
  4. UTILITY   Would a downstream label survive?  Same LLM classifies original and synthetic on an eDiscovery
               taxonomy; agreement / Cohen's kappa = label preservation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from email.utils import format_datetime
from datetime import datetime

import numpy as np

from .llm import LLM
from .parsing import ParsedEmail, parse_email
from .pii import clean_display_name, name_from_address
from .personas import guess_gender

_GROUP_WORDS = {'the', 'company', 'bank', 'department', 'management', 'relations', 'review', 'family', 'group', 'team',
                'services', 'trading', 'desk', 'risk', 'office', 'committee', 'all', 'list', 'energy', 'corp', 'inc'}
from . import prompts

JUDGE_MODEL = "anthropic/claude-haiku-4.5"

GLOBAL_PATTERNS = {
    "enron": re.compile(r"enron", re.I),
    "houston": re.compile(r"\bhouston\b", re.I),
    "unit_acronym": re.compile(r"\b(?:ENA|ECT|EES|EWS|EGM|EBS|EIM|EOL|HPL|NNG)\b"),
    "lotus_notes": re.compile(r"/[A-Z]{2,5}/[A-Z]{2,5}(?:@[A-Za-z]+)?"),
    "enron_scandal_terms": re.compile(r"\b(?:Raptor|LJM|Chewco|JEDI|Fastow|Skilling|Andersen)\b"),
}


# ----------------------------------------------------------------------------------------- helpers
def render_real(e: ParsedEmail) -> str:
    """Render an original e-mail with the *same* header template as the synthetic ones (fair judging)."""
    def nm(addr: str, x: str = "") -> str:
        n = clean_display_name(x) or name_from_address(addr).replace(".", " ").title()
        return f"{n} <{addr}>"
    date = e.date and format_datetime(datetime.fromisoformat(e.date))
    lines = [f"Date: {date}", f"From: {nm(e.from_addr, e.x_from)}", f"To: {', '.join(nm(a) for a in e.to_addrs)}"]
    if e.cc_addrs:
        lines.append(f"Cc: {', '.join(nm(a) for a in e.cc_addrs)}")
    lines += [f"Subject: {e.subject}", "Mime-Version: 1.0", "Content-Type: text/plain; charset=us-ascii", "", e.body]
    if e.trailer:
        lines += ["", e.trailer]
    return "\n".join(lines)


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def ngrams(words: list[str], n: int) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def structure(body: str, trailer: str, n_to: int, n_cc: int, subject: str) -> dict:
    b = body.strip()
    return {
        "body_words": len(b.split()),
        "n_lines": len([l for l in b.split("\n") if l.strip()]),
        "n_paragraphs": len([p for p in re.split(r"\n\s*\n", b) if p.strip()]),
        "has_greeting": bool(re.match(r"\s*(hi|hello|dear|hey|all|team|good (morning|afternoon))\b", b[:60], re.I)),
        "has_signoff": bool(re.search(r"(thanks|thank you|regards|best|sincerely|cheers)[,!.]?\s*(\n|$)", b[-250:], re.I)),
        "has_trailer": bool(trailer.strip()),
        "is_re": subject.lower().startswith("re:"),
        "is_fw": subject.lower().startswith(("fw:", "fwd:")),
        "n_to": n_to, "n_cc": n_cc,
        "all_lower": b == b.lower() and len(b) > 20,
        "has_url": bool(re.search(r"https?://|www\.", b)),
        "mentions_attachment": bool(re.search(r"\battach", b, re.I)),
    }


# ----------------------------------------------------------------------------------------- privacy
def privacy_row(orig: ParsedEmail, syn: dict, people_fullnames: set[str]) -> dict:
    synth_names = {n.lower() for n in syn["meta"].get("synthetic_people", [])}
    text = syn["raw"]
    words = toks(syn["body"] + " " + syn["quoted"])
    ow = toks(orig.body + " " + orig.trailer)
    syn_ng, or_ng = ngrams(words, 8), ngrams(ow, 8)
    copy8 = len(syn_ng & or_ng) / max(len(syn_ng), 1)
    own = set(toks(syn["headers"].get("From", "") + " " + syn["headers"].get("To", "") + " " + syn["headers"].get("Cc", "")))
    low = text.lower()
    hits = {k: bool(p.search(text)) for k, p in GLOBAL_PATTERNS.items()}
    person_hits = [n for n in people_fullnames if n in low                       # cheap substring test first
                   and re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", low)
                   and not all(t in own for t in n.split()) and n not in synth_names]
    return {
        "verifier_hard_leak": bool(syn["meta"].get("leaks_after")),
        "verifier_leaks": syn["meta"].get("leaks_after", []),
        "forced_redaction": bool(syn["meta"].get("forced_redaction")),
        "verbatim_8gram_overlap": copy8,
        "global_leak": any(hits.values()),
        "global_hits": [k for k, v in hits.items() if v],
        "enron_person_fullname": bool(person_hits),
        "enron_person_hits": person_hits[:3],
        "any_leak": bool(syn["meta"].get("leaks_after")) or any(hits.values()) or bool(person_hits),
        "placeholder_left": bool(re.search(r"\[(?:COMPANY|ORG_\d+|LOC_\d+|UNIT_\d+)\]", text)),
    }


def build_people_fullnames(raw_messages, min_len: int = 5) -> set[str]:
    """Full names (lower-case, 'first last') of every person appearing in From/To headers of a corpus sample."""
    names: Counter = Counter()
    for m in raw_messages:
        e = parse_email(m)
        cands = [clean_display_name(e.x_from)] + [name_from_address(a).replace(".", " ") for a in e.to_addrs + e.cc_addrs]
        for c in cands:
            t = c.lower().split()
            if len(t) >= 2 and all(x.isalpha() and len(x) >= 3 for x in (t[0], t[-1])) and guess_gender(t[0]) != "u" \
                    and not (set(t) & _GROUP_WORDS):
                names[f"{t[0]} {t[-1]}"] += 1
    return {n for n in names if len(n) >= min_len}


# ----------------------------------------------------------------------------------------- fidelity
def fidelity_frame(orig_list: list[ParsedEmail], syn_list: list[dict]):
    import pandas as pd
    rows = []
    for o, s in zip(orig_list, syn_list):
        so = structure(o.body, o.trailer, len(o.to_addrs), len(o.cc_addrs), o.subject)
        ss = structure(s["body"], s["quoted"], len(re.findall(r"<[^>]+>", s["headers"].get("To", ""))),
                       len(re.findall(r"<[^>]+>", s["headers"].get("Cc", ""))), s["subject"])
        rows.append({**{f"o_{k}": v for k, v in so.items()}, **{f"s_{k}": v for k, v in ss.items()}})
    return pd.DataFrame(rows)


def fidelity_summary(df) -> dict:
    from scipy import stats
    out = {}
    ratio = (df["s_body_words"] + 1) / (df["o_body_words"] + 1)
    out["body_words_ratio_median"] = float(ratio.median())
    out["body_words_ratio_iqr"] = [float(ratio.quantile(.25)), float(ratio.quantile(.75))]
    out["body_words_spearman"] = float(stats.spearmanr(df["o_body_words"], df["s_body_words"])[0])
    out["body_words_ks_stat"] = float(stats.ks_2samp(df["o_body_words"], df["s_body_words"])[0])
    out["body_words_ks_p"] = float(stats.ks_2samp(df["o_body_words"], df["s_body_words"])[1])
    for f in ["has_greeting", "has_signoff", "has_trailer", "is_re", "is_fw", "all_lower", "has_url", "mentions_attachment"]:
        out[f"{f}_agree"] = float((df[f"o_{f}"] == df[f"s_{f}"]).mean())
        out[f"{f}_orig_rate"] = float(df[f"o_{f}"].mean())
        out[f"{f}_syn_rate"] = float(df[f"s_{f}"].mean())
    for f in ["n_paragraphs", "n_lines"]:
        out[f"{f}_spearman"] = float(stats.spearmanr(df[f"o_{f}"], df[f"s_{f}"])[0])
    out["n_to_exact_agree"] = float((df["o_n_to"] == df["s_n_to"]).mean())
    return out


def distinct_n(texts: list[str], n: int) -> float:
    grams = []
    for t in texts:
        w = toks(t)
        grams += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


# ----------------------------------------------------------------------------------------- LLM judging
class Judge:
    def __init__(self, model: str = JUDGE_MODEL):
        self.llm = LLM(model, max_cost=25.0)

    def realism(self, email_text: str) -> dict:
        out = self.llm.chat([{"role": "system", "content": prompts.JUDGE_SYSTEM},
                             {"role": "user", "content": prompts.JUDGE_REALISM.format(email=email_text[:6000])}],
                            temperature=0.0, max_tokens=300)
        return _json(out)

    def labels(self, email_text: str) -> dict:
        out = self.llm.chat([{"role": "system", "content": prompts.JUDGE_SYSTEM},
                             {"role": "user", "content": prompts.JUDGE_LABELS.format(email=email_text[:6000])}],
                            temperature=0.0, max_tokens=200)
        return _json(out)

    def pairwise(self, real_text: str, syn_text: str, real_first: bool) -> dict:
        a, b = (real_text, syn_text) if real_first else (syn_text, real_text)
        out = self.llm.chat([{"role": "system", "content": prompts.JUDGE_SYSTEM},
                             {"role": "user", "content": prompts.JUDGE_PAIRWISE.format(a=a[:5000], b=b[:5000])}],
                            temperature=0.0, max_tokens=200)
        j = _json(out)
        picked_real = j.get("real")
        truth = "A" if real_first else "B"
        return {"correct": picked_real == truth, "picked": picked_real, "reason": j.get("reason", "")}


def cohen_kappa(a: list, b: list) -> float:
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b), key=str)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
