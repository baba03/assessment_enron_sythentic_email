"""Create a blank human-review sheet (CSV) with 30 stratified original/synthetic pairs.

    python scripts/make_review_sheet.py [arm]          -> reports/human_review_sheet.csv
Reviewer fills the last six columns (see reports/human_review_README.md), then:
    python scripts/summarize_review.py                 -> mean scores + agreement with the automatic metrics
"""
import csv, json, random, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
arm = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
ev = pd.read_json(ROOT / "data" / "eval_set.jsonl", lines=True)
run = {d["idx"]: d for d in map(json.loads, open(ROOT / "data" / "runs" / f"{arm}.jsonl")) if "error" not in d}
random.seed(3)
idx = sorted(random.sample(sorted(run), 30))
def body(m):  # original body without the X-* header block
    return m.split("X-FileName:")[1].split("\n", 1)[1].strip() if "X-FileName:" in m else m
with open(ROOT / "reports" / "human_review_sheet.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "original_excerpt", "synthetic_email", "realism_1to5", "original_identifier_left_YN", "intent_preserved_YN",
                "format_realistic_1to5", "sender_signature_consistent_YN", "notes"])
    for i in idx:
        w.writerow([i, body(ev.iloc[i]["message"])[:900], run[i]["raw"][:1400], "", "", "", "", "", ""])
print("wrote", len(idx), "rows")
