"""Command line entry points.

  # one e-mail in, one synthetic e-mail out (the core deliverable)
  python -m enron_synth.cli one --input tests/sample_email.txt
  python -m enron_synth.cli one --input tests/sample_email.txt --company kaveribank --json
  cat email.txt | python -m enron_synth.cli one

  # batch: build a de-duplicated synthetic dataset from emails.csv (or the cached sample)
  python -m enron_synth.cli dataset --csv ../emails.csv --n 500 --out data/synthetic.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .personas import COMPANY_BY_ID
from .pipeline import DEFAULT_MODEL, SyntheticEmailGenerator


def cmd_one(a):
    raw = Path(a.input).read_text() if a.input else sys.stdin.read()
    gen = SyntheticEmailGenerator(model=a.model, temperature=a.temperature)
    out = gen.generate(raw, file=a.file or "", company=a.company, mode=a.mode)
    if a.json:
        print(json.dumps(out.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(out.raw)
        m = out.meta
        print(f"\n--- company={out.company} model={m['model']} attempts={m['attempts']} leaks={m['leaks_after']} "
              f"forced_redaction={m['forced_redaction']} scrub={m['scrub_counts']}", file=sys.stderr)


def _body_key(raw: str) -> str:
    body = raw.split("\n\n", 1)[-1]
    return hashlib.md5(re.sub(r"\s+", " ", body).strip().lower().encode()).hexdigest()


def cmd_dataset(a):
    import pandas as pd
    from tqdm import tqdm
    from .parsing import parse_email
    gen = SyntheticEmailGenerator(model=a.model, temperature=a.temperature)
    rows = []
    seen: set[str] = set()
    for chunk in pd.read_csv(a.csv, chunksize=20000):
        chunk = chunk.sample(frac=min(1.0, a.n * 6 / 517401 * 20), random_state=a.seed)   # cheap over-sample, then filter
        for f, m in zip(chunk["file"], chunk["message"]):
            k = _body_key(m)
            if k in seen:
                continue                      # 52 % of Enron rows are body-duplicates: never synthesise the same text twice
            e = parse_email(m, f)
            if not (a.min_words <= len(e.body.split()) <= a.max_words) or not e.to_addrs or not e.date:
                continue
            seen.add(k)
            rows.append((f, m))
        if len(rows) >= a.n:
            break
    rows = rows[: a.n]

    def work(x):
        f, m = x
        try:
            d = gen.generate(m, f).to_dict()
            d["source_file"] = f
            return d
        except Exception as ex:  # noqa: BLE001
            return {"source_file": f, "error": repr(ex)}

    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "w") as fh:
        for d in tqdm(ex.map(work, rows), total=len(rows)):
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")   # attempts / retries go to stderr
    ap = argparse.ArgumentParser(prog="enron_synth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one", help="synthesise a single e-mail")
    o.add_argument("--input", help="file with the raw Enron e-mail (headers + body); default stdin")
    o.add_argument("--file", help="original path in emails.csv, e.g. allen-p/_sent_mail/1. (used to pick company / date shift)")
    o.add_argument("--company", choices=sorted(COMPANY_BY_ID), help="target company (default: deterministic per mailbox)")
    o.add_argument("--model", default=DEFAULT_MODEL)
    o.add_argument("--mode", choices=["hybrid", "rule", "llm_only"], default="hybrid")
    o.add_argument("--temperature", type=float, default=0.7)
    o.add_argument("--json", action="store_true")
    o.set_defaults(fn=cmd_one)
    d = sub.add_parser("dataset", help="batch-generate a de-duplicated synthetic dataset")
    d.add_argument("--csv", required=True)
    d.add_argument("--n", type=int, default=200)
    d.add_argument("--out", default="data/synthetic.jsonl")
    d.add_argument("--model", default=DEFAULT_MODEL)
    d.add_argument("--temperature", type=float, default=0.7)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--min-words", type=int, default=8)
    d.add_argument("--max-words", type=int, default=700)
    d.set_defaults(fn=cmd_dataset)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
