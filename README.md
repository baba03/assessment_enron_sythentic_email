# Synthetic e-mail generation from the Enron corpus (Consilio Labs technical assessment)

A prototype that takes **one real Enron e-mail** and returns a **synthetic e-mail that looks like it came from a different company**, with the original people, organisations, contact details and Enron-specific traces removed, while keeping the structure, tone and intent that downstream LLM fine-tuning tasks (classification, privilege / sensitivity detection, thread reconstruction ...) depend on.

| Deliverable | Where |
|---|---|
| 1. Experiment plan (≤ ½ page) | [docs/01_experiment_plan.md](docs/01_experiment_plan.md) |
| 2. Results & findings (≤ 1 page) | [docs/02_results_and_findings_1page.md](docs/02_results_and_findings_1page.md) (strict one page) · [full version](docs/02_results_and_findings.md) |
| Additional (offline) evaluation + manual audit | [docs/05_additional_evaluation_and_audit.md](docs/05_additional_evaluation_and_audit.md) |
| Failure analysis · scale-up & cost · edge cases | [docs/03_failure_analysis.md](docs/03_failure_analysis.md) · [docs/04_scale_up_cost_and_edge_cases.md](docs/04_scale_up_cost_and_edge_cases.md) |
| Open items (credit, GitHub, references) | [NEXT_STEPS.md](NEXT_STEPS.md) |
| 3. Prototype code | [src/enron_synth/](src/enron_synth/) · CLI · [notebooks/01_demo.ipynb](notebooks/01_demo.ipynb) |
| Supporting analysis | [notebooks/02_analysis.ipynb](notebooks/02_analysis.ipynb) · [reports/](reports/) |

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env            # put your OpenRouter key in OPENROUTER_API_KEY

pip install -e .                # needs pip >= 21.3; otherwise skip this line and use: export PYTHONPATH=src

# one e-mail in -> one synthetic e-mail out
python -m enron_synth.cli one --input tests/sample_email.txt
python -m enron_synth.cli one --input tests/sample_email.txt --company kaveribank --json
python -m enron_synth.cli one --input tests/sample_email.txt --mode rule     # no LLM / no API key needed
```

The first command prints the synthetic e-mail (headers + body + any quoted history) and, on stderr, a one-line report: target company, model, LLM attempts, residual leaks, whether the deterministic repair had to fire.

From Python:

```python
from enron_synth.pipeline import SyntheticEmailGenerator
gen = SyntheticEmailGenerator()                       # default model: openai/gpt-4.1-mini via OpenRouter
out = gen.generate(raw_enron_email_text, file="allen-p/_sent_mail/1.")   # `file` (the emails.csv path) picks the target company
print(out.raw)          # RFC-822-style text: this is the training example
out.subject, out.body, out.quoted, out.headers, out.meta
```

`mode="rule"` (no LLM, no API key needed) and `mode="llm_only"` are the two baselines used in the ablation.

Batch a dataset (de-duplicated by body, because 52.7 % of `emails.csv` rows are body-duplicates):

```bash
PYTHONPATH=src python -m enron_synth.cli dataset --csv ../emails.csv --n 500 --out data/synthetic.jsonl
```

## Target companies: Indian only (simple)

The synthetic e-mails are set in **12 fictional Indian companies** (agriculture, power, banking, logistics, telecom, hospitals, IT services, food, steel, municipal utility, law firm, investment) in different Indian cities: Indian names, `Rs.` with lakh / crore, `+91` phone numbers, `.in` domains, IST time zone, English text. The earlier experiments (stored in `data/runs/`) were run with a wider multi-country list; use `SYNTH_COMPANY_SET=international` only to reproduce those stored outputs. Regenerating with the Indian set needs API credit (~US$0.15 for the 120 e-mails).

## Results snapshot (120 stratified e-mails; details in [docs/02](docs/02_results_and_findings.md))

| Arm | Any leak ↓ | Verbatim 8-gram copy ↓ | Length ρ ↑ | Structure agree ↑ | Judge spots synthetic ↓ |
|---|---|---|---|---|---|
| Naive LLM only | 55.8 % | 5.3 % | 0.41 | 73.5 % | 83 % |
| Rule-based only | 11.7 % | 52 % | 1.00 | 99.8 % | 69 % |
| **Hybrid v1** | 6.7 % | 10.4 % | 0.99 | 99.0 % | 63 % |
| **Hybrid v2** | 3.3 % (4.2 % strict) | 3.1 % | 0.99 | 99.4 % | not yet judged (form classifier AUC 0.69) |

Full tables: [reports/results_offline.md](reports/results_offline.md), [reports/results_judged.md](reports/results_judged.md); figures in `reports/figures/`.

### Status: what is finished, what is not

* Done: EDA on all 517k e-mails (two passes), the pipeline (v2.2), 16 offline tests, embedding / classifier / consistency / diversity evaluation, generation for all arms, privacy/fidelity metrics for every arm, LLM-judged realism/label metrics for the **v1** arms.
* **Not done - the OpenRouter account ran out of credit** during the final evaluation (`/api/v1/credits`: 390.19 used of 390; this project used US$3.2). Pending: (1) regenerate the 40/120 hybrid outputs whose scrubbed text changed with the final name-harvesting fixes, (2) LLM-judge hybrid v2, (3) LLM-judge the four alternative generator models. Once the account has credit: `sh scripts/finish_evaluation.sh` (~US$2). Everything already computed is cached in `data/cache/`, so it reproduces offline with `SYNTH_OFFLINE=1`.

## How it works

```
raw Enron e-mail
   │  parsing.py      headers / new text / forwarded-or-quoted trailer
   ▼
STAGE 1  pii.py  - deterministic de-identification, ONE regex pass
   │  people (header identities, "Last, First" quoted headers, NER, name-lexicon safety net, nicknames -> ONE person)
   │  e-mail addresses / phones / URLs / long ids  -> Indian-format fakes (+91 phones, .in / .co.in domains)
   │  absolute dates in the text                   -> shifted by whole weeks (weekday + clock time preserved)
   │  Enron-isms: company, unit acronyms, known Enron-world orgs, Lotus-Notes "/HOU/ECT@ECT", mailto blobs
   │  everything the LLM must *invent* (orgs, places, company)  -> [ORG_1] [LOC_2] [COMPANY] placeholders
   ▼
STAGE 2  pipeline.py + prompts.py  - constrained LLM rewrite (OpenRouter)
   │  input: scrubbed text (the LLM never sees the original people) + target-company profile + style fingerprint
   │  job:   transfer the business domain, fill placeholders, keep intent / tone / structure / length / sensitivity cues
   ▼
VERIFY + REPAIR
   │  leak detector (per-e-mail hard terms + famous Enron surnames), placeholder remnants, length drift,
   │  name preservation, missing quoted block  -> retry with concrete feedback (≤ 3 attempts)
   │  still leaking?  -> deterministic forced redaction
   ▼
render: Message-ID / Date (tz of the target city) / From / To / Cc / Subject + body + quoted history,
        original hard-wrap width re-applied
```

Design choices worth knowing (details in the write-up):

* **Consistency across e-mails.** The target company is chosen from the Enron *mailbox*; the synthetic person is a keyed hash of the original identity. Same Enron person ⇒ same synthetic person in every e-mail of that mailbox (threads stay coherent). Set `salt` to make the mapping irreversible.
* **The LLM only sees pseudonymised text.** In production this stage can run on a self-hosted open-weight model (Llama / Qwen worked in the comparison) so real client data never leaves the boundary.
* **Defence in depth for PII**: deterministic scrub → LLM told to replace survivors → independent verifier → forced redaction → separate *evaluation-time* leak checks that use different signals (see `evaluate.py`).

## Reproducing the experiments

```bash
export PYTHONPATH=src
python -m enron_synth.eda --csv ../emails.csv               # full-corpus EDA (~12 min) -> reports/eda_summary.json, figures
python -m enron_synth.dupes ../emails.csv                    # duplicate analysis -> reports/dup_stats.json
python -m enron_synth.experiment select --n 120              # stratified evaluation set
python -m enron_synth.experiment run --arm hybrid            # arms: hybrid | rule | llm_only  (--model, --name)
python -m enron_synth.experiment evaluate --arms hybrid rule llm_only --no-judge --out results_offline.json   # privacy + fidelity, no API
python -m enron_synth.experiment evaluate --arms hybrid rule llm_only --out results_judged.json               # + LLM judge (needs credit)
python -m enron_synth.evaluate_extra --arms hybrid rule llm_only   # embeddings, classifier, consistency (offline)
python -m enron_synth.report                                 # tables + figures
python scripts/make_review_sheet.py                          # blank human-review sheet (30 pairs)
python -m pytest tests -q                                    # offline unit tests (no API key)
```

All LLM responses are cached under `data/cache/`, so re-running is free and deterministic. The OpenRouter key is read from `.env` (git-ignored). `LLM.report()` prints calls / cache hits / dollars spent; the whole study cost only a few dollars.

> **Rate limit note.** On the key used for this assessment OpenRouter returned HTTP 402 `in_flight_budget_exhausted` above ~2 concurrent requests, so `llm.py` caps concurrency globally (`SYNTH_MAX_CONCURRENCY`, default 2) and backs off on 402/429.

## Repository layout

```
src/enron_synth/  parsing.py  pii.py  enron_terms.py  personas.py  prompts.py  llm.py  pipeline.py
                  evaluate.py  experiment.py  eda.py  dupes.py  report.py  cli.py
tests/            offline unit tests + the brief's example e-mail
notebooks/        01_demo.ipynb (function)   02_analysis.ipynb (EDA + results)
docs/             experiment plan, results & findings
reports/          eda_summary.json, dup_stats.json, results*.json, tables, figures
data/             eval_set.jsonl, runs/*.jsonl (generated outputs of every arm)   [big parquet files git-ignored]
```

## Known limitations

* No formal privacy guarantee: this is empirical de-identification with independent checks, not differential privacy. Names that are unusual, mis-spelled or only implied can survive both stages; the verifier catches what it knows about (and famous Enron surnames), not everything.
* Cross-mailbox threads are not linked (different mailbox ⇒ different target company), and dates are shifted per mailbox.
* Rare, very long e-mails are truncated (body 1 200 words, quoted history 450 words); attachments are not part of the dataset.
* Short generic / automated mails are only lightly re-worded (high verbatim overlap), which is harmless for PII but matters if memorisation is a concern.
* LLM-as-judge metrics (realism, label preservation) use one judge model; treat absolute values with care and use them for comparing arms.
