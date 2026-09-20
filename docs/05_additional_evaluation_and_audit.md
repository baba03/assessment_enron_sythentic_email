# 5. Additional evaluation (no API needed) and manual audit

Everything here is computed offline (`python -m enron_synth.evaluate_extra`, `eda_extra`), so it covers the final hybrid v2 outputs that the LLM judge could not (no credit). n = 120 e-mails.

## 5.1 More data analysis (all 517,401 e-mails; `reports/eda_extra.json`, `eda_vocab.json`)
* **Who:** 20,325 unique senders, 78,383 unique recipients, 5,291 sender / 11,626 recipient domains; 83 % of mail is sent from `enron.com`. 65 % of mail is internal-only, 16 % arrives from external senders, 15 % is internal-to-external.
* **Threads:** 127k distinct normalised subjects; 67 % occur in more than one message (median 2, p95 11). Threads are therefore common; a fine-tuning set with quoted history teaches them.
* **Time:** peak month Oct 2001 (37k). Only 4 % of mail is sent on weekends. **Data-quality issue:** 605 e-mails (0.12 %) have impossible dates (timestamps from 1979 to 2044), so the date shift must not assume clean dates (the code falls back to a default).
* **Content:** top terms are `enron, ect, energy, power, gas, market, california, company, business`; top bigrams include `hou ect`, `enron enronxgate`, `natural gas`, `north america`, `vice president`. Some bodies contain HTML (`href`, `font`, `image`): **0.5 %** of e-mails; 6.7 % have bullet lists, 8.3 % table-like columns, 2.3 % exceed 1,200 words (truncated by the pipeline).

## 5.2 Results

| Arm | Embedding cos (pair / random) | Own-source top-1 / top-5 | Form-classifier AUC raw → **styled** ↓ | Source names surviving ↓ | Sender name in sign-off | distinct-2 |
|---|---|---|---|---|---|---|
| *Original Enron (reference)* | - | - | 0.50 by definition | - | 46 % | 0.765 |
| Naive LLM only | 0.83 / 0.61 | 92 % / 97 % | 0.96 → 0.96 | 3.3 % | **0 %** | 0.723 |
| Rule-based, scrubber as of v2 | 0.90 / 0.62 | 100 / 100 % | 0.77 | 5.0 % (22 names) | 44 % | 0.777 |
| Rule-based, **final** scrubber | 0.90 / 0.62 | 100 / 100 % | 0.77 | **2.5 % (3 names)** | 41 % | 0.778 |
| Hybrid v1 | 0.79 / 0.61 | 88 % / 96 % | 0.86 → **0.68** | 5.8 % (46 names) | 41 % | 0.801 |
| **Hybrid v2** (as run) | 0.80 / 0.61 | 89 % / 98 % | 0.89 → **0.69** | 4.2 % (21 names) | 44 % | 0.807 |

(Alternative generators, same pipeline: form AUC 0.82-0.87 → 0.64-0.71 after style restoration; full table in `reports/results_extra.md`.)

**Findings**
1. **Semantic fidelity holds despite the domain change.** The synthetic e-mail is closer to its own source than to any of the other 119 originals 89 % of the time (98 % within the top 5; chance is < 1 %); pair cosine 0.80 vs 0.61 for random pairs. The rule-based arm is 100 % because it does not change the content, which is exactly the problem (52 % verbatim copy).
2. **A real-vs-synthetic classifier (form features only, no vocabulary; 5×10-fold CV) separates hybrid v2 from real mail with AUC 0.89.** Diagnosis: the LLM "cleans" three things the source has: trailing spaces on Lotus-wrapped lines (19 % of source lines vs 0 %), curly quotes / long dashes (208 non-ASCII characters vs 0), and double spaces after full stops (65 % of sentence breaks vs 2 %). `restore_style()` reapplies them deterministically after the LLM; applied to the stored outputs (no LLM calls) AUC fell to **0.69** (Qwen 0.64). The residual was mostly accented names and £ / € / ₹ symbols from the earlier multi-country company set; the default set is now Indian-only with `Rs.`, so this residual largely disappears (not re-measured with an LLM run). The naive LLM stays at 0.96.
3. **Stage-1 name detection improved measurably**: names from the source's recipient lists that survive in the output fell from 22 (5.0 % of e-mails) with the v2 scrubber to 3 (2.5 %) with the final one. The stored hybrid v2 outputs (4.2 %, 21 names) predate this fix. This metric is stricter than the earlier "3.3 %" figure because it also catches names with middle initials and line wraps; the earlier figure was therefore optimistic.
4. **The naive LLM cannot keep header and body consistent** (sender's name appears in the sign-off 0 % of the time vs 46 % in the source); hybrid 44 % (rule-based 41 %) matches the source.
5. **Diversity is preserved**: distinct-2 0.81 vs 0.77 for the source; every e-mail's company is derived from the mailbox, giving 13 different companies (in the earlier run, which used a multi-country set; the default is now 12 Indian companies).

## 5.3 Manual audit (performed by the AI assistant, not a human; a blank sheet for a human is in `reports/human_review_sheet.csv`)
I read 18 stored hybrid v2 outputs side by side with their sources (two rounds: 10 during development, 8 random at the end). What worked: domain transfer that keeps the argument (negawatts → renewable-energy credits; energy-market memo → agribusiness), consistent people across header/body/quote, correct weekday-preserving dates, local phone formats, believable invented organisations. Defects found:

| Sample | Defect | Status |
|---|---|---|
| 111 | Houston ZIP `77002` survived in "Charlotte, NC 77002"; `x@dynegy.com@ENRON` artefact | **Fixed** (ZIP + double-@ rules, test) |
| 42 | File name kept the source date (`Manifesto … 1-25-01.doc`) | **Fixed** (dash-separated dates shifted, test) |
| 117 | Names with a middle initial / line wrap in a Lotus recipient list survived (`Teresa G Bushman`, `Harry M Collins`) | **Detected** by the new survivor metric; final scrubber removes most (3 remaining in 120), stored output needs regeneration |
| 98 | Quoted-printable debris `=20` in output | Fixed in final scrubber; stored output stale |
| 55 | A private note about a drill and a house key became a workplace "vault access / safe deposit box" message (personal→business drift) | **Open** (prompt rule exists, not sufficient); needs API to re-test |
| several | Lotus `/HOU/Fleet Operations@Fleet Operations` fragments in long recipient lists | Reduced by final scrubber; partly stale |

## 5.4 Human review (to be done by a person)
`reports/human_review_sheet.csv` (30 stratified pairs) and `reports/human_review_README.md` define a 30-minute protocol; `python scripts/summarize_review.py` summarises it. Not yet filled.
