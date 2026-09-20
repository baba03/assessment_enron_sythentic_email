"""Apply the deterministic style restoration (pipeline.restore_style) to STORED outputs - no LLM calls.

    python scripts/postprocess_run.py hybrid hybrid_v1 ...     ->  data/runs/<arm>_styled.jsonl
"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enron_synth.parsing import parse_email
from enron_synth.pipeline import restore_style, _ASCII_FOLD

ROOT = Path(__file__).resolve().parents[1]
ev = pd.read_json(ROOT / "data" / "eval_set.jsonl", lines=True)
for arm in sys.argv[1:]:
    out = []
    for d in map(json.loads, open(ROOT / "data" / "runs" / f"{arm}.jsonl")):
        if "error" in d:
            continue
        o = parse_email(ev.iloc[d["idx"]]["message"], ev.iloc[d["idx"]]["file"])
        d["subject"] = d["subject"].translate(_ASCII_FOLD)
        d["body"], d["quoted"] = restore_style(d["body"], o.body), restore_style(d["quoted"], o.trailer)
        h = dict(d["headers"]); h["Subject"] = d["subject"]; d["headers"] = h
        head = "\n".join(f"{k}: {v}" for k, v in h.items())
        cs = "us-ascii" if (head + d["body"] + d["quoted"]).isascii() else "utf-8"
        d["raw"] = f"{head}\nMime-Version: 1.0\nContent-Type: text/plain; charset={cs}\n\n{d['body']}" + (f"\n\n{d['quoted']}" if d["quoted"].strip() else "")
        out.append(d)
    with open(ROOT / "data" / "runs" / f"{arm}_styled.jsonl", "w") as fh:
        for d in out:
            fh.write(json.dumps(d) + "\n")
    print(arm, "->", len(out))
