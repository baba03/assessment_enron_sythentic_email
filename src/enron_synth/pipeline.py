"""End-to-end generator:  raw Enron e-mail  ->  synthetic e-mail from a fictional company.

    parse -> scrub (deterministic PII removal) -> LLM domain transfer -> verify (leaks / format) -> repair -> render

Three modes exist so the design can be ablated:
    hybrid    (proposed)  scrub + constrained LLM rewrite + verifier loop
    rule      scrub only; placeholders filled from Faker (no LLM)                  - "classic" de-identification
    llm_only  raw e-mail straight to an LLM with a naive instruction                - "just ask the LLM"
"""
from __future__ import annotations

import hashlib
import os
import re
import statistics
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.utils import format_datetime
from zoneinfo import ZoneInfo

import logging

from .llm import LLM
from .parsing import ParsedEmail, parse_email
from .personas import COMPANIES, COMPANY_BY_ID, Company, PersonaFactory, Person, pick_company, _h
from .pii import Scrubbed, Scrubber, find_leaks
from . import prompts

DEFAULT_MODEL = os.getenv("SYNTH_MODEL", "openai/gpt-4.1-mini")

TZ = {"agriindia": "Asia/Kolkata", "nordwind": "Europe/Berlin", "halcyon": "Europe/London", "tasman": "Australia/Perth",
      "brasilog": "America/Sao_Paulo", "maple": "America/Edmonton", "lumiere": "Europe/Paris",
      "cedarpoint": "America/New_York", "orion": "Europe/Dublin", "solmar": "Europe/Madrid",
      "kestrel": "Europe/London", "lakeshore": "America/New_York", "meridian": "Europe/London"}

log = logging.getLogger("enron_synth")

_PLACEHOLDER_RE = re.compile(r"\[(?:COMPANY|ORG_\d+|LOC_\d+|UNIT_\d+)\]")


@dataclass
class SyntheticEmail:
    raw: str                     # full RFC-822-style text: headers + body (+ quoted block)  <- the training example
    subject: str
    body: str                    # new text written by the sender
    quoted: str                  # forwarded / replied-to history
    headers: dict
    company: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------ date shifting
def shift_date(iso: str | None, owner: str, company: Company, salt: str) -> tuple[datetime, int]:
    """Shift by a whole number of WEEKS into 2024-25: weekday + wall-clock time of day are preserved
    (business-hours distribution intact) and every message of one mailbox moves by the same amount
    (relative timing / thread order intact)."""
    try:
        dt = datetime.fromisoformat(iso).replace(tzinfo=None) if iso else datetime(2001, 6, 1, 10, 0)
    except ValueError:
        dt = datetime(2001, 6, 1, 10, 0)
    target_year = 2024 + (_h(owner, salt=salt) % 2)
    days = int(round((target_year - dt.year) * 365.25 / 7)) * 7
    new = dt + timedelta(days=days)
    return new.replace(tzinfo=ZoneInfo(TZ.get(company.id, "Asia/Kolkata" if company.locale == "en_IN" else "UTC"))), days


# ------------------------------------------------------------------------------------ style fingerprint
def style_fingerprint(body: str) -> dict:
    lines = [l for l in body.split("\n") if l.strip()]
    lens = sorted(len(l) for l in lines) or [0]
    p90 = lens[int(0.9 * (len(lens) - 1))]
    med = statistics.median(lens)
    wrapped = len(lines) >= 4 and p90 <= 82 and med >= 40
    return {"n_words": len(body.split()), "wrapped": wrapped, "wrap_width": p90}


def wrap_profile(text: str) -> int | None:
    """Return the hard-wrap width of `text` if it looks hard-wrapped (typical of 2001-era plain-text mail), else None."""
    lines = [l for l in text.split("\n") if l.strip() and not l.lstrip().startswith((">", "http", "www"))]
    long_ = sorted(len(l) for l in lines if len(l) >= 50)
    if len(lines) < 3 or len(long_) < 2 or long_[-1] > 90:
        return None
    return max(60, min(80, long_[int(0.9 * (len(long_) - 1))]))


def rewrap(text: str, width: int | None) -> str:
    """Re-apply hard wrapping to over-long lines only (keeps blank lines, short lines, signatures, lists)."""
    if not width or not text:
        return text
    import textwrap
    out = []
    for line in text.split("\n"):
        if len(line) > width and not line.lstrip().startswith(("http", "www", "|")):
            indent = re.match(r"\s*", line).group(0)
            out += textwrap.wrap(line, width=width, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False) or [line]
        else:
            out.append(line)
    return "\n".join(out)


def unwrap(text: str, width: int | None) -> str:
    """Join hard-wrapped continuation lines so the LLM sees paragraphs, not 75-column fragments."""
    if not width or not text:
        return text
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        cur = lines[i]
        while (i + 1 < len(lines) and lines[i + 1].strip() and len(cur.rstrip()) >= width - 14 and cur.strip()
               and not re.match(r"\s*([-*\u2022>|]|\d+[.)])\s", lines[i + 1]) and not cur.rstrip().endswith((":", "---"))
               and not re.match(r"\s*(From|To|Cc|cc|Sent|Subject|Date)\s*:", lines[i + 1])):
            cur = cur.rstrip() + " " + lines[i + 1].strip()
            i += 1
        out.append(cur)
        i += 1
    return "\n".join(out)


_ASCII_FOLD = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...",
                              "\u00a0": " ", "\u2022": "-"})


def restore_style(text: str, ref: str) -> str:
    """Re-apply typographic habits of the SOURCE that LLMs systematically 'clean up' (found via the real-vs-synthetic classifier):
       * typographic quotes / dashes -> ASCII (the source corpus is pure ASCII)
       * double space after sentence end, if the source mostly does that
       * trailing space on wrapped lines (Lotus-Notes wrapping), at the source's rate
    Deterministic (hash-based, no RNG state), so results are reproducible."""
    if not text:
        return text
    text = text.translate(_ASCII_FOLD)
    two = len(re.findall(r"[.!?]  [A-Z]", ref))
    one = len(re.findall(r"[.!?] [A-Z]", ref))
    if two + one >= 3 and two / (two + one) >= 0.4:
        text = re.sub(r"([.!?]) (?=[A-Z])", r"\1  ", text)
    lines = text.split("\n")
    cand = [i for i, l in enumerate(lines[:-1]) if len(l) >= 50 and lines[i + 1].strip() and l.strip()]
    rl = ref.split("\n")
    rcand = [i for i, l in enumerate(rl[:-1]) if len(l.rstrip()) >= 50 and rl[i + 1].strip()]
    rate = (sum(rl[i].endswith(" ") for i in rcand) / len(rcand)) if len(rcand) >= 3 else 0.0
    if rate >= 0.2:
        for i in cand:
            if int(hashlib.md5(lines[i].encode()).hexdigest(), 16) % 100 < rate * 100:
                lines[i] = lines[i].rstrip() + " "
    return "\n".join(lines)


def _tail(text: str, k: int = 4) -> str:
    return " ".join([l for l in text.strip().split("\n") if l.strip()][-k:]).lower()


def signature_problem(scrubbed_body: str, out_body: str, sender_first: str) -> str | None:
    """If the source was signed by the sender (their name is in the last lines), the synthetic e-mail must be too."""
    f = sender_first.lower()
    if f and f in _tail(scrubbed_body) and f not in _tail(out_body):
        return f"The sign-off must name the sender ({sender_first}); the input is signed by the sender, keep the same signer."
    return None


def _tags(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.S)
    return m.group(1).strip("\n") if m else None


def _fmt_addr(p: Person) -> str:
    return f"{p.full} <{p.email}>"


class SyntheticEmailGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, salt: str = "synth-v1", max_cost: float = 15.0,
                 use_ner: bool = True, companies: list[Company] | None = None, temperature: float = 0.7):
        self.model = model
        self.salt = salt
        self.use_ner = use_ner
        self.companies = companies or COMPANIES
        self.temperature = temperature
        self._llm = None
        self.max_cost = max_cost
        self._factories: dict[str, PersonaFactory] = {}
        self._lock = threading.RLock()   # Faker instances are shared + re-seeded => serialise the (fast) deterministic parts

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM(self.model, max_cost=self.max_cost)
        return self._llm

    def factory(self, company: Company) -> PersonaFactory:
        # one factory per company => the same Enron person maps to the same synthetic person across emails
        if company.id not in self._factories:
            self._factories[company.id] = PersonaFactory(company, self.salt)
        return self._factories[company.id]

    # ------------------------------------------------------------------ public API
    def generate(self, email: str | ParsedEmail, file: str = "", company: Company | str | None = None,
                 mode: str = "hybrid", max_attempts: int = 3, seed: str = "") -> SyntheticEmail:
        """`email` is the raw Enron message text (headers + body) or an already parsed ParsedEmail."""
        e = email if isinstance(email, ParsedEmail) else parse_email(email, file)
        owner = (e.file or file).split("/")[0] or e.x_origin or e.from_addr
        if isinstance(company, str):
            company = COMPANY_BY_ID[company]
        company = company or pick_company(owner, self.salt, self.companies)
        new_dt, shift_days = shift_date(e.date, owner, company, self.salt)
        with self._lock:
            f = self.factory(company)
            s = Scrubber(f, use_ner=self.use_ner).scrub(e, shift_days=shift_days)
        fp = style_fingerprint(e.body)

        meta = {"mode": mode, "model": self.model if mode != "rule" else None, "attempts": 0, "leaks_after": [],
                "forced_redaction": False, "scrub_counts": s.counts, "shift_days": shift_days,
                "orig_words": fp["n_words"], "trailer_truncated": len(e.trailer.split()) > 450}

        if mode == "rule":
            subject, body, quoted = self._rule_fill(s, f)
        elif mode == "llm_only":
            subject, body, quoted = self._llm_only(e, company, seed)
            meta["attempts"] = 1
        else:
            wb, wt = wrap_profile(e.body), wrap_profile(e.trailer)
            s.body, s.trailer = unwrap(s.body, wb), unwrap(s.trailer, wt)
            subject, body, quoted = self._hybrid(s, e, company, new_dt, shift_days, fp, meta, max_attempts, seed)
            body, quoted = rewrap(body, wb), rewrap(quoted, wt)

        # last line of defence: nothing on the hard list may leave the building (skipped for llm_only baseline)
        if mode != "llm_only":
            subject, body, quoted, forced = self._force_redact(subject, body, quoted, s, f)
            meta["forced_redaction"] = forced
        leaks = find_leaks("\n".join([subject, body, quoted]), s)
        meta["synthetic_people"] = sorted(s.synthetic_names)
        meta["leaks_after"] = [l.term for l in leaks if l.severity == "hard"]
        meta["soft_leaks_after"] = [l.term for l in leaks if l.severity == "soft"]

        return self._assemble(s, e, company, new_dt, subject, body, quoted, meta)

    # ------------------------------------------------------------------ hybrid
    def _hybrid(self, s: Scrubbed, e: ParsedEmail, company: Company, new_dt, shift_days, fp, meta, max_attempts, seed):
        n = max(fp["n_words"], 1)
        lo, hi = int(n * 0.7), int(n * 1.35) + 5
        old_dt = datetime.fromisoformat(e.date) if e.date else new_dt
        wrap_hint = "Do NOT hard-wrap paragraphs (wrapping is re-applied automatically afterwards)."
        placeholder_list = ", ".join(sorted(s.placeholders)) or "none"
        feedback = ""
        best = None
        for attempt in range(1, max_attempts + 1):
            user = prompts.USER_HYBRID.format(
                company_name=company.name, company_short=company.short, industry=company.industry, city=company.city,
                country=company.country, currency_note=company.currency_note, units=", ".join(company.sub_units),
                domain=company.domain, sender_name=s.from_person.full, sender_title=s.from_person.title,
                wrap_hint=wrap_hint, n_words=n, lo=lo, hi=hi,
                placeholder_list=placeholder_list, new_date=new_dt.strftime("%A, %d %B %Y"), shift_days=shift_days,
                new_year=new_dt.year,
                subject=s.subject, body=s.body, trailer=s.trailer, feedback=feedback)
            out = self.llm.chat([{"role": "system", "content": prompts.SYSTEM_HYBRID}, {"role": "user", "content": user}],
                                temperature=self.temperature, max_tokens=3500, cache_salt=f"{seed}|{attempt}")
            meta["attempts"] = attempt
            subject, body, quoted = _tags(out, "subject"), _tags(out, "body"), _tags(out, "quoted") or ""
            problems = []
            if subject is None or body is None:
                problems.append("Output did not follow the <subject>/<body>/<quoted> format.")
                subject, body = subject or s.subject, body or ""
            else:
                text = "\n".join([subject, body, quoted])
                if _PLACEHOLDER_RE.search(text):
                    problems.append("Unreplaced placeholders remain: " + ", ".join(sorted(set(_PLACEHOLDER_RE.findall(text)))) + ".")
                hard = [l.term for l in find_leaks(text, s) if l.severity == "hard"]
                if hard:
                    problems.append("Identifying original terms still present (replace them with invented ones): " + ", ".join(hard) + ".")
                w = len(body.split())
                if n >= 25 and not (0.5 * n <= w <= 1.8 * n):
                    problems.append(f"Body length is {w} words but the original is {n}; aim for {lo}-{hi}.")
                sp = signature_problem(s.body, body, s.from_person.first)
                if sp:
                    problems.append(sp)
                low = text.lower()
                lost = [n for n in s.keep_names if n.lower() not in low]
                if lost:
                    problems.append("You changed or dropped these people's names, which must be kept exactly as given: " + ", ".join(lost[:8]) + ".")
                if s.trailer.strip() and not quoted.strip():
                    problems.append("The original had a quoted/forwarded block; keep one.")
            if problems:
                log.info("attempt %d/%d rejected (%s): %s", attempt, max_attempts, company.id, "; ".join(p[:80] for p in problems))
            cand = (subject, body, quoted)
            if best is None or len(problems) < best[0]:
                best = (len(problems), cand)
            if not problems:
                return cand
            feedback = prompts.FEEDBACK_TMPL.format(problems="\n".join(f"- {p}" for p in problems))
        return best[1]

    # ------------------------------------------------------------------ baselines
    def _rule_fill(self, s: Scrubbed, f: PersonaFactory):
        with self._lock:
            return self._rule_fill_locked(s, f)

    def _rule_fill_locked(self, s: Scrubbed, f: PersonaFactory):
        c = s.company

        def fill(t: str) -> str:
            def r(m):
                tag = m.group(0)
                if tag == "[COMPANY]":
                    return c.short
                kind = tag[1:-1].split("_")[0]
                if kind == "ORG":
                    return f.org_name(tag)
                if kind == "LOC":
                    return f.city_name(tag)
                return c.sub_units[_h(tag) % len(c.sub_units)] if c.sub_units else c.short
            return _PLACEHOLDER_RE.sub(r, t)
        return fill(s.subject), fill(s.body), fill(s.trailer)

    def _llm_only(self, e: ParsedEmail, company: Company, seed: str):
        user = prompts.USER_LLM_ONLY.format(
            company_name=company.name, industry=company.industry, country=company.country,
            from_=e.from_addr, to=", ".join(e.to_addrs), subject=e.subject, body=e.body,
            trailer=("\n" + e.trailer) if e.trailer else "")
        out = self.llm.chat([{"role": "system", "content": prompts.SYSTEM_LLM_ONLY}, {"role": "user", "content": user}],
                            temperature=self.temperature, max_tokens=3500, cache_salt=f"{seed}|llm_only")
        return _tags(out, "subject") or e.subject, _tags(out, "body") or out, ""

    # ------------------------------------------------------------------ verification / repair
    def _force_redact(self, subject, body, quoted, s: Scrubbed, f: PersonaFactory):
        """Deterministic repair for anything still on the hard list after the LLM retries."""
        with self._lock:
            return self._force_redact_locked(subject, body, quoted, s, f)

    def _force_redact_locked(self, subject, body, quoted, s: Scrubbed, f: PersonaFactory):
        forced = False
        c = s.company
        joined = "\n".join([subject, body, quoted])
        leaks = [l.term for l in find_leaks(joined, s) if l.severity == "hard"]
        if not leaks:
            return subject, body, quoted, False

        def repl_for(term: str) -> str:
            if term in ("enron",) or term.startswith("enron"):
                return c.short
            if term == "houston" or term.startswith("houston"):
                return c.city
            if " " in term and "@" not in term and not term.isdigit():
                return f.person_for(term).full if term.replace(" ", "").isalpha() and len(term.split()) == 2 else f.org_name(term)
            if "@" in term:
                return f.person_for(term.split("@")[0]).email
            if term.isdigit():
                return f.fresh_phone_like(term)
            return f.person_for(term).last if term.isalpha() else f.org_name(term)

        def fix(t: str) -> str:
            for term in sorted(leaks, key=len, reverse=True):
                t = re.sub(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", repl_for(term), t, flags=re.I)
            return t
        forced = True
        return fix(subject), fix(body), fix(quoted), forced

    # ------------------------------------------------------------------ rendering
    @staticmethod
    def _tidy(t: str) -> str:
        return "\n".join(l.rstrip() for l in t.split("\n")).strip("\n")

    def _assemble(self, s: Scrubbed, e: ParsedEmail, company: Company, new_dt, subject, body, quoted, meta) -> SyntheticEmail:
        subject = subject.strip().translate(_ASCII_FOLD)
        body, quoted = restore_style(self._tidy(body), e.body), restore_style(self._tidy(quoted), e.trailer)
        with self._lock:
            return self._assemble_locked(s, e, company, new_dt, subject, body, quoted, meta)

    def _assemble_locked(self, s, e, company, new_dt, subject, body, quoted, meta) -> SyntheticEmail:
        mid = hashlib.md5((e.message_id + company.id + self.salt).encode()).hexdigest()
        headers = {
            "Message-ID": f"<{mid[:16]}.{int(mid[16:24], 16) % 100000}@{company.domain}>",
            "Date": format_datetime(new_dt),
            "From": _fmt_addr(s.from_person),
            "To": ", ".join(_fmt_addr(p) for p in s.to_people),
        }
        if s.cc_people:
            headers["Cc"] = ", ".join(_fmt_addr(p) for p in s.cc_people)
        headers["Subject"] = subject
        head = "\n".join(f"{k}: {v}" for k, v in headers.items() if v != "" or k == "Subject")
        charset = "us-ascii" if (head + body + quoted).isascii() else "utf-8"
        head += f"\nMime-Version: 1.0\nContent-Type: text/plain; charset={charset}"
        raw = f"{head}\n\n{body}" + (f"\n\n{quoted}" if quoted.strip() else "")
        return SyntheticEmail(raw=raw, subject=subject, body=body, quoted=quoted, headers=headers,
                              company=company.id, meta=meta)
