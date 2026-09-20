"""Summarise the filled-in reports/human_review_sheet.csv."""
from pathlib import Path
import pandas as pd
df = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "human_review_sheet.csv")
done = df.dropna(subset=["realism_1to5"])
print(f"{len(done)}/{len(df)} rows reviewed")
if len(done):
    yn = lambda c: (done[c].astype(str).str.upper().str.startswith("Y")).mean()
    print(f"realism mean {done['realism_1to5'].mean():.2f} | format realism mean {done['format_realistic_1to5'].mean():.2f}")
    print(f"original identifier left: {yn('original_identifier_left_YN'):.0%} | intent preserved: {yn('intent_preserved_YN'):.0%} | signature consistent: {yn('sender_signature_consistent_YN'):.0%}")
