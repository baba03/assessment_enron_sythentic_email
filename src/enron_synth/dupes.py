"""Duplicate analysis: how much of emails.csv is the same message stored in several folders?"""
import hashlib, json, re
from collections import Counter
from pathlib import Path
import pandas as pd

MID = re.compile(r"Message-ID: (<[^>]+>)")
ROOT = Path(__file__).resolve().parents[2]

def run(csv):
    mid_c, body_c, folder_dups = Counter(), Counter(), Counter()
    first_folder = {}
    n = 0
    for chunk in pd.read_csv(csv, chunksize=50000):
        for f, m in zip(chunk["file"], chunk["message"]):
            n += 1
            mm = MID.search(m)
            mid = mm.group(1) if mm else f
            mid_c[mid] += 1
            body = m.split("\n\n", 1)[-1]
            body_c[hashlib.md5(re.sub(r"\s+", " ", body).encode()).hexdigest()] += 1
            folder = f.split("/")[1] if "/" in f else ""
            if mid in first_folder:
                folder_dups[folder] += 1
            else:
                first_folder[mid] = folder
    out = {"n_rows": n, "n_unique_message_id": len(mid_c), "pct_rows_duplicate_by_message_id": 1 - len(mid_c) / n,
           "n_unique_body": len(body_c), "pct_rows_duplicate_by_body": 1 - len(body_c) / n,
           "max_copies_of_one_message_id": max(mid_c.values()),
           "duplicate_rows_by_folder_top10": dict(folder_dups.most_common(10))}
    (ROOT / "reports" / "dup_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    import sys; run(sys.argv[1])
