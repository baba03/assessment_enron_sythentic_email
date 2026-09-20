"""Full-corpus EDA over emails.csv (streamed in chunks - the file is 1.4 GB).

Outputs (in data/ and reports/figures/):
  data/features.parquet   one row per email with structural features
  data/sample.parquet     random 30k-email sample with parsed fields (input for experiments)
  reports/eda_summary.json
  reports/figures/*.png

Usage:  python -m enron_synth.eda --csv ../emails.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .parsing import parse_email, find_emails

PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|mm|k|m|b)?", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
ATTACH_RE = re.compile(r"\b(attach(?:ed|ment)|\.xls|\.doc|\.pdf|\.ppt)\b", re.I)
GREETING_RE = re.compile(r"^\s*(hi|hello|dear|hey|all|team|good (morning|afternoon))\b", re.I)
SIGNOFF_RE = re.compile(r"(thanks|thank you|regards|best|sincerely|cheers)[,!.]?\s*$", re.I | re.M)
ENRON_RE = re.compile(r"enron", re.I)


def featurize(p, raw_len: int) -> dict:
    body_words = len(p.body.split())
    return {
        "file": p.file,
        "folder_type": p.folder_type,
        "owner": p.file.split("/")[0],
        "date": p.date,
        "n_to": len(p.to_addrs),
        "n_cc": len(p.cc_addrs),
        "subject_len": len(p.subject.split()),
        "is_re": p.subject.lower().startswith("re:"),
        "is_fw": p.subject.lower().startswith(("fw:", "fwd:")),
        "empty_subject": not p.subject,
        "body_words": body_words,
        "trailer_words": len(p.trailer.split()),
        "has_trailer": bool(p.trailer),
        "n_lines": p.body.count("\n") + 1,
        "n_emails_in_text": len(find_emails(p.raw_body)),
        "n_phones": len(PHONE_RE.findall(p.raw_body)),
        "n_money": len(MONEY_RE.findall(p.raw_body)),
        "n_urls": len(URL_RE.findall(p.raw_body)),
        "mentions_attachment": bool(ATTACH_RE.search(p.body)),
        "has_greeting": bool(GREETING_RE.search(p.body[:60])),
        "has_signoff": bool(SIGNOFF_RE.search(p.body[-200:])),
        "mentions_enron": bool(ENRON_RE.search(p.raw_body)),
        "external_sender": not p.from_addr.endswith("@enron.com"),
        "raw_len": raw_len,
    }


def run(csv_path: str, out_dir: str, chunksize: int = 20000, sample_n: int = 30000, seed: int = 7):
    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    feats, samples = [], []
    total_rows = 517401  # known size, only used for the progress bar
    with tqdm(total=total_rows, unit="email") as bar:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            keep_p = sample_n / total_rows
            for f, m in zip(chunk["file"], chunk["message"]):
                p = parse_email(m, f)
                feats.append(featurize(p, len(m)))
                if rng.random() < keep_p:
                    samples.append({"file": f, "message": m})
            bar.update(len(chunk))

    df = pd.DataFrame(feats)
    df.to_parquet(out / "data" / "features.parquet")
    pd.DataFrame(samples).to_parquet(out / "data" / "sample.parquet")
    summarize(df, out)


def summarize(df: pd.DataFrame, out: Path):
    fig_dir = out / "reports" / "figures"
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.year
    q = lambda s: {k: float(v) for k, v in s.quantile([.05, .25, .5, .75, .95, .99]).items()}

    summary = {
        "n_emails": int(len(df)),
        "n_mailboxes": int(df["owner"].nunique()),
        "folder_type_top15": df["folder_type"].value_counts().head(15).to_dict(),
        "body_words_quantiles": q(df["body_words"]),
        "body_words_mean": float(df["body_words"].mean()),
        "pct_empty_body": float((df["body_words"] == 0).mean()),
        "pct_with_forward_or_quote": float(df["has_trailer"].mean()),
        "pct_reply": float(df["is_re"].mean()),
        "pct_forward": float(df["is_fw"].mean()),
        "pct_empty_subject": float(df["empty_subject"].mean()),
        "n_to_quantiles": q(df["n_to"]),
        "pct_multi_recipient": float((df["n_to"] > 1).mean()),
        "pct_with_cc": float((df["n_cc"] > 0).mean()),
        "pct_mentions_attachment": float(df["mentions_attachment"].mean()),
        "pct_greeting": float(df["has_greeting"].mean()),
        "pct_signoff": float(df["has_signoff"].mean()),
        "pct_mentions_enron": float(df["mentions_enron"].mean()),
        "pct_external_sender": float(df["external_sender"].mean()),
        "pct_with_phone": float((df["n_phones"] > 0).mean()),
        "pct_with_money": float((df["n_money"] > 0).mean()),
        "pct_with_url": float((df["n_urls"] > 0).mean()),
        "pct_with_embedded_email_addr": float((df["n_emails_in_text"] > 0).mean()),
        "year_counts": {str(int(k)): int(v) for k, v in df["year"].value_counts().sort_index().items() if 1995 <= k <= 2003},
    }
    (out / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figures
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(np.log10(df["body_words"] + 1), bins=50, color="#4c72b0")
    ax[0, 0].set(title="Body length (log10 words+1)", xlabel="log10(words+1)")
    df["folder_type"].value_counts().head(10).sort_values().plot.barh(ax=ax[0, 1], color="#55a868")
    ax[0, 1].set(title="Top folders")
    df["n_to"].clip(upper=15).value_counts().sort_index().plot.bar(ax=ax[0, 2], color="#c44e52")
    ax[0, 2].set(title="# recipients (clipped 15)")
    yc = df["year"].value_counts().sort_index()
    yc = yc[(yc.index >= 1997) & (yc.index <= 2002)]
    yc.plot.bar(ax=ax[1, 0], color="#8172b2")
    ax[1, 0].set(title="Emails per year")
    flags = ["has_trailer", "is_re", "is_fw", "mentions_attachment", "has_greeting", "has_signoff", "mentions_enron"]
    df[flags].mean().sort_values().plot.barh(ax=ax[1, 1], color="#ccb974")
    ax[1, 1].set(title="Structural feature prevalence")
    pii = pd.Series({
        "email addr in text": (df["n_emails_in_text"] > 0).mean(),
        "phone": (df["n_phones"] > 0).mean(),
        "money": (df["n_money"] > 0).mean(),
        "url": (df["n_urls"] > 0).mean(),
        "'Enron' word": df["mentions_enron"].mean(),
    }).sort_values()
    pii.plot.barh(ax=ax[1, 2], color="#64b5cd")
    ax[1, 2].set(title="Identifier prevalence (fraction of emails)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_overview.png", dpi=130)
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    run(a.csv, a.out)
