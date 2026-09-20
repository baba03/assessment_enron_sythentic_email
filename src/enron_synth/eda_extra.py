"""Second EDA pass (header-level, fast): who talks to whom, when, and how threads look.

    python -m enron_synth.eda_extra --csv ../emails.csv     ->  reports/eda_extra.json, reports/figures/eda_extra.png
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from email.utils import parsedate_to_datetime

ROOT = Path(__file__).resolve().parents[2]
ADDR = re.compile(r"[\w.+'-]+@([\w-]+(?:\.[\w-]+)+)")
HDR = {k: re.compile(rf"^{k}:[ \t]*(.*(?:\n[ \t]+.*)*)", re.M) for k in ("From", "To", "Cc", "Date", "Subject")}
PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.I)


def run(csv: str):
    senders, rcpts, sdom, rdom, subj = Counter(), Counter(), Counter(), Counter(), Counter()
    months, hours, dows = Counter(), Counter(), Counter()
    n = internal_only = external_in = external_out = 0
    for chunk in pd.read_csv(csv, chunksize=50000):
        for m in chunk["message"]:
            head = m.split("\n\n", 1)[0]
            n += 1
            g = {k: (r.search(head).group(1) if r.search(head) else "") for k, r in HDR.items()}
            fa = ADDR.findall(g["From"])
            fs = re.findall(r"[\w.+'-]+@[\w.-]+", g["From"])
            ta = re.findall(r"[\w.+'-]+@[\w.-]+", g["To"] + "," + g["Cc"])
            if fs:
                senders[fs[0].lower()] += 1
                sdom[fs[0].split("@")[1].lower()] += 1
            for a in ta:
                rcpts[a.lower()] += 1
                rdom[a.split("@")[1].lower()] += 1
            if fs and ta:
                s_int = fs[0].lower().endswith("enron.com")
                r_int = all(a.lower().endswith("enron.com") for a in ta)
                internal_only += s_int and r_int
                external_in += (not s_int)
                external_out += s_int and not r_int
            try:
                d = parsedate_to_datetime(g["Date"])
                months[d.strftime("%Y-%m")] += 1
                hours[d.hour] += 1
                dows[d.weekday()] += 1
            except (TypeError, ValueError):
                pass
            subj[PREFIX.sub("", g["Subject"]).strip().lower()] += 1
    threads = {k: v for k, v in subj.items() if k}
    sizes = pd.Series(list(threads.values()))
    out = {
        "n_emails": n, "unique_senders": len(senders), "unique_recipients": len(rcpts),
        "unique_sender_domains": len(sdom), "unique_recipient_domains": len(rdom),
        "top_sender_domains": sdom.most_common(10), "top_senders": senders.most_common(10),
        "pct_internal_only": internal_only / n, "pct_from_external_sender": external_in / n, "pct_internal_to_external": external_out / n,
        "date_range": [min(months), max(months)], "busiest_month": months.most_common(1)[0],
        "hour_of_day_pct": {h: hours[h] / sum(hours.values()) for h in sorted(hours)},
        "weekday_pct": {d: dows[d] / sum(dows.values()) for d in sorted(dows)},
        "unique_normalised_subjects": len(threads), "pct_subjects_with_>1_message": float((sizes > 1).mean()),
        "median_messages_per_subject": float(sizes.median()), "p95_messages_per_subject": float(sizes.quantile(.95)),
        "pct_empty_subject": subj[""] / n,
    }
    (ROOT / "reports" / "eda_extra.json").write_text(json.dumps(out, indent=2, default=str))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ms = pd.Series(months).sort_index()
    ms = ms[(ms.index >= "1999-01") & (ms.index <= "2002-08")]
    ax[0].plot(range(len(ms)), ms.values); ax[0].set_xticks(range(0, len(ms), 6), ms.index[::6], rotation=45, fontsize=8); ax[0].set_title("E-mails per month")
    ax[1].bar(list(out["hour_of_day_pct"]), list(out["hour_of_day_pct"].values())); ax[1].set_title("Hour of day (sender local time)")
    ax[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], [out["weekday_pct"].get(i, 0) for i in range(7)]); ax[2].set_title("Day of week")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "eda_extra.png", dpi=120)
    print(json.dumps({k: v for k, v in out.items() if k not in ("hour_of_day_pct", "weekday_pct")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); run(ap.parse_args().csv)
