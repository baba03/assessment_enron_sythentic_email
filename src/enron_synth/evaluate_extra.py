"""Offline (no API) evaluation that complements the LLM judge.

  1. SEMANTIC FIDELITY   local embeddings (BGE-small via fastembed): cosine(original, synthetic) for the pair vs. for random
                         other pairs, and top-1 retrieval (is the synthetic e-mail closest to its own source?).
  2. REALISM (classifier) can a classifier tell real from synthetic using FORM only (no vocabulary)? AUC 0.5 = cannot.
  3. CONSISTENCY         sender name in the sign-off, greeting name = recipient, header/domain consistency, vs. the originals.
  4. DIVERSITY           mean pairwise similarity inside the synthetic set vs. inside the originals; distinct-n.

    python -m enron_synth.evaluate_extra --arms hybrid_v1 hybrid rule llm_only ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import distinct_n
from .parsing import parse_email
from .pii import clean_display_name

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs"
_EMB = None


_CACHE_PATH = ROOT / "data" / "embed_cache.pkl"


def embed(texts: list[str]) -> np.ndarray:
    """Local BGE-small embeddings, cached on disk. NB: the default fastembed batch size hangs on long e-mails -> batch_size=16."""
    global _EMB
    import hashlib
    import pickle
    cache = pickle.loads(_CACHE_PATH.read_bytes()) if _CACHE_PATH.exists() else {}
    keys = [hashlib.sha1(t[:1200].encode()).hexdigest() for t in texts]
    todo = [(k, t[:1200]) for k, t in zip(keys, texts) if k not in cache]
    if todo:
        if _EMB is None:
            from fastembed import TextEmbedding
            _EMB = TextEmbedding("BAAI/bge-small-en-v1.5")
        for (k, _), v in zip(todo, _EMB.embed([t for _, t in todo], batch_size=16)):
            cache[k] = np.asarray(v, dtype=np.float32)
        _CACHE_PATH.write_bytes(pickle.dumps(cache))
    v = np.array([cache[k] for k in keys])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def form_features(body: str, quoted: str) -> list[float]:
    """Content-free features: length, layout, character classes, punctuation, wrapping. No words."""
    t = body.strip()
    lines = t.split("\n")
    ne = [l for l in lines if l.strip()]
    ll = [len(l) for l in ne] or [0]
    n = max(len(t), 1)
    words = t.split()
    return [np.log1p(len(words)), np.log1p(len(ne)), np.mean(ll), np.percentile(ll, 90), max(ll),
            sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n, sum(c in ".,;:" for c in t) / n,
            t.count("!") / n * 100, t.count("?") / n * 100, sum(ord(c) > 127 for c in t) / n * 100,
            sum(l.endswith(" ") for l in lines) / max(len(lines), 1), t.count("  ") / n * 100,
            (len(lines) - len(ne)) / max(len(lines), 1), float(bool(quoted.strip())), np.mean([len(w) for w in words]) if words else 0,
            float(bool(re.match(r"\s*(hi|hello|dear|hey)\b", t[:40], re.I))), float(t == t.lower() and len(t) > 20),
            float(bool(re.search(r"(thanks|regards|best|sincerely)[,!.]?\s*$", t[-120:], re.I | re.M))),
            sum(len(l) > 80 for l in ne) / max(len(ne), 1), float(bool(re.search(r"https?://|www\.", t)))]


def discriminator_auc(real_X: np.ndarray, syn_X: np.ndarray, seed: int = 0) -> tuple[float, float]:
    X = np.vstack([real_X, syn_X])
    y = np.array([0] * len(real_X) + [1] * len(syn_X))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def consistency(e, d: dict | None) -> dict:
    """Same checks on the original (d=None) and on the synthetic e-mail."""
    if d is None:
        body = e.body
        sender_first = (clean_display_name(e.x_from).split() or [""])[0].lower()
        to_first = [(clean_display_name(x).split() or [""])[0].lower() for x in re.split(r",\s*", e.x_to)] if e.x_to else []
    else:
        body = d["body"]
        m = re.match(r"(.+?) <", d["headers"]["From"])
        sender_first = m.group(1).split()[0].lower() if m else ""
        to_first = [x.split()[0].lower() for x in re.findall(r"([^,<]+?) <", d["headers"].get("To", ""))]
    tail = " ".join(body.strip().split("\n")[-3:]).lower()
    has_sign = bool(re.search(r"(regards|thanks|thank you|best|sincerely)", body[-200:].lower()))
    g = re.match(r"\s*(?:hi|hello|dear|hey)\s+([A-Za-z]+)", body.strip(), re.I)
    return {"sign_ok": (sender_first in tail) if has_sign and sender_first else None,
            "greet_ok": (g.group(1).lower() in to_first) if g and to_first else None}


_LOTUS = re.compile(r"([A-Z][A-Za-z'.-]+)((?:\s+[A-Z]\.?)?)\s+([A-Z][A-Za-z'-]+)\s*/\s*(?:[A-Z]{2,5}|Corp|NA)\b")
_LASTFIRST = re.compile(r"(?m)^[ \t>]*(?:From|To|Cc|cc|Sent by):[ \t]*\"?([A-Z][A-Za-z'-]+),\s+([A-Z][A-Za-z'-]+)")


def source_names(o) -> set[tuple[str, str]]:
    """(first, last) of people the SOURCE e-mail names in Lotus-Notes lists ('Teresa G\\nBushman/HOU/ECT') or 'Last, First' headers.
    Independent of the scrubber: its own regexes, tolerant to middle initials and line wraps."""
    text = o.body + "\n" + o.trailer
    out = {(m.group(1).lower(), m.group(3).lower()) for m in _LOTUS.finditer(text)}
    out |= {(m.group(2).lower(), m.group(1).lower()) for m in _LASTFIRST.finditer(text)}
    return {n for n in out if len(n[0]) >= 3 and len(n[1]) >= 3}


def survivors(names: set[tuple[str, str]], out_text: str, synthetic: set[str]) -> list[str]:
    flat = re.sub(r"\s+", " ", out_text.lower())
    hits = []
    for f, l in names:
        if f"{f} {l}" in synthetic:
            continue
        if re.search(rf"(?<![a-z]){re.escape(f)}(?: [a-z]\.?)? {re.escape(l)}(?![a-z])", flat):
            hits.append(f"{f} {l}")
    return hits


def run(arms: list[str]):
    ev = pd.read_json(ROOT / "data" / "eval_set.jsonl", lines=True)
    origs = [parse_email(m, f) for m, f in zip(ev["message"], ev["file"])]
    o_txt = [o.subject + "\n" + o.body for o in origs]
    Eo = embed(o_txt)
    real_X = np.array([form_features(o.body, o.trailer) for o in origs])
    co = [consistency(o, None) for o in origs]
    rate = lambda rows, k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
    out = {"source_emails_with_named_people_in_lists": float(np.mean([len(source_names(o)) > 0 for o in origs])),
           "originals": {"pairwise_similarity_within": float((Eo @ Eo.T)[np.triu_indices(len(Eo), 1)].mean()),
                         "distinct2": distinct_n([o.body for o in origs], 2), "distinct1": distinct_n([o.body for o in origs], 1),
                         "sign_ok": rate(co, "sign_ok"), "greet_ok": rate(co, "greet_ok")}}
    rng = np.random.default_rng(0)
    for arm in arms:
        run_ = {d["idx"]: d for d in map(json.loads, open(RUNS / f"{arm}.jsonl")) if "error" not in d}
        idx = sorted(run_)
        syn = [run_[i] for i in idx]
        Es = embed([d["subject"] + "\n" + d["body"] for d in syn])
        Eo_i = Eo[idx]
        sim = (Es * Eo_i).sum(1)
        S = Es @ Eo_i.T
        rand = np.array([S[i, rng.permutation(len(idx))[0]] for i in range(len(idx))])
        sync = np.array([form_features(d["body"], d["quoted"]) for d in syn])
        auc, sd = discriminator_auc(real_X[idx], sync)
        cs = [consistency(origs[i], d) for i, d in zip(idx, syn)]
        surv = [survivors(source_names(origs[i]), d["raw"], {n.lower() for n in d["meta"].get("synthetic_people", [])}) for i, d in zip(idx, syn)]
        out[arm] = {
            "n": len(idx),
            "embed_cos_pair_mean": float(sim.mean()), "embed_cos_random_pair_mean": float(S.mean()),
            "embed_top1_retrieval": float((S.argmax(1) == np.arange(len(idx))).mean()),
            "embed_top5_retrieval": float(np.mean([i in np.argsort(-S[i])[:5] for i in range(len(idx))])),
            "form_discriminator_auc": auc, "form_discriminator_auc_sd": sd,
            "sign_ok": rate(cs, "sign_ok"), "greet_ok": rate(cs, "greet_ok"),
            "source_name_survivor_email_rate": float(np.mean([len(x) > 0 for x in surv])),
            "source_name_survivors_total": int(sum(len(x) for x in surv)),
            "source_name_survivor_examples": [x[:3] for x in surv if x][:4],
            "pairwise_similarity_within": float((Es @ Es.T)[np.triu_indices(len(Es), 1)].mean()),
            "distinct1": distinct_n([d["body"] for d in syn], 1), "distinct2": distinct_n([d["body"] for d in syn], 2),
            "n_companies": len({d["company"] for d in syn}),
        }
        print(arm, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[arm].items()})
    (ROOT / "reports" / "results_extra.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", required=True)
    run(ap.parse_args().arms)
