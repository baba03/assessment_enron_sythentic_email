# data/

| Path | What | In git? |
|---|---|---|
| `eval_set.jsonl` | The 120 stratified Enron e-mails used in all experiments (raw text) | yes |
| `runs/*.jsonl` | Generated outputs of every arm (hybrid v1/v2, rule, llm_only, four alternative models) | yes |
| `cache/` | Cached OpenRouter responses (generator + judge). Lets everything reproduce offline with `SYNTH_OFFLINE=1` | yes (~9 MB) |
| `sample.parquet`, `features.parquet` | 30 k-e-mail sample and per-e-mail features from the full corpus | **no** (git-ignored, regenerate) |
| `emails.csv` | The Kaggle Enron corpus (1.4 GB, 517,401 rows) | **no** (not included) |

Regenerate the ignored files: `PYTHONPATH=src python -m enron_synth.eda --csv /path/to/emails.csv`
(~12 minutes; also writes `reports/eda_summary.json` and the EDA figure).
