# 2. Experiment results and findings

> **Status.** Generation, privacy and fidelity results are complete. LLM-judged results (realism, labels) exist only for the **v1** pipeline: the OpenRouter *account* balance ran out mid-evaluation (`/credits`: 390.19 used of 390; my usage was US$3.2; the key's own US$40 limit was not the cause). v2 is verified by offline metrics (embedding retrieval, real-vs-synthetic classifier, name-survivor check; see `docs/05`), tests and manual audit; `scripts/finish_evaluation.sh` completes the rest (~US$2).

## Data analysis (all 517,401 e-mails; `notebooks/02_analysis.ipynb`)
Mail is short and informal: median body **45 words** (p95 515), 8.6 % empty, greeting in 5 %, sign-off in 18 %, 73 % with ≤1 recipient. **36 % carry quoted/forwarded history** (Lotus-Notes/Outlook formats). Identifiers are everywhere: 48 % mention "Enron", 31 % embed e-mail addresses, 24 % phone numbers. Enron-isms betray the source even without names (`/HOU/ECT@ECT`, `EOL`, mailto blobs, 75-column wrap, quoted-printable debris). **52.7 % of rows are body-duplicates** (245k distinct bodies) → dedupe before sampling.

## Synthesis process
*PII removal (stage 1, deterministic, one regex pass):* header identities (nickname-aware), `"Last, First" <addr>` pairs, spaCy persons, a given-name-lexicon net (capitalised, lower-case, line-wrapped), addresses, phones, ids and Enron-isms; only orgs/places/company become placeholders for the LLM. Persons are a keyed hash → same Enron person = same synthetic person across a mailbox. *Distribution preservation:* one of 12 fictional Indian companies (Indian names, Rs./lakh/crore, +91 phones, English text) per mailbox; dates shifted by whole weeks (weekday, clock time, gaps and in-text dates preserved); length, wrapping, quoted history and greeting/sign-off constrained or restored deterministically. *Stage 2* LLM transfers the domain; *verify/repair* retries (≤3), then forces redaction.

## Results (n = 120; 95 % Wilson CI)
| Arm | Any leak ↓ | 8-gram copy ↓ | Length ρ ↑ | Structure ↑ | Judge spots synthetic ↓ | Category κ ↑ |
|---|---|---|---|---|---|---|
| Naive LLM only | **55.8 %** [47–64] | 5.3 % | 0.41 | 73.5 % | 83 % [76–89] | 0.80 |
| Rule-based only | 11.7 % [7–19] | **52 %** | 1.00 | 99.8 % | 69 % [60–77] | 0.82 |
| **Hybrid v1** | 6.7 % [3–13] | 10.4 % | 0.99 | 99.0 % | **63 %** [54–71] | 0.69 |
| **Hybrid v2** | **3.3 %** [1–8] (4.2 % by the stricter name check) | 3.1 % | 0.99 | 99.4 % | not judged (classifier AUC 0.69 after style fix, was 0.89) | not judged (top-1 own-source retrieval 89 %) |

Naive LLM leaves original identifiers in over half the e-mails and drops the quoted history in every e-mail that had one; rules alone leave Enron's business context (52 % verbatim, 11.7 % still with Enron markers). Hybrid is best on leakage *and* fidelity (differences vs naive are large; v1-vs-v2 and hybrid-vs-rule are within CI, i.e. directional). **Realism is not solved**: the judge still spots the synthetic e-mail 63 % of the time (chance 50 %; real mail scores 4.0/5, hybrid 2.7). Labels survive well: intent 87 %, privilege 96 %, action 95 %, sensitive 89 %; *category* κ 0.69 because v1 sometimes turned personal mail into work mail. Other generators (v1 pipeline): Llama-3.3-70B (open-weight, US$0.0004/e-mail) leaks 3.3 %, Qwen/Gemini-Flash/Haiku 5.8 %; all keep structure (ρ ≥ 0.96) but copy more source text (25-30 %) → open-weight models are viable on-prem.

## Challenges
1. **The independent evaluator found leaks my verifier could not**: names in Lotus-Notes recipient lists (double spaces, line wraps, `janel= =20 guerrero`, all-lower-case writers). Fixed (whitespace/QP/lexicon harvesting), regression-tested (13 tests) and confirmed on all 4 leaking e-mails at stage 1. **Caveat:** 40/120 hybrid outputs need regeneration with the final scrubber (needs API); 3.3 % is the pre-fix figure and a lower bound (measured against a 30k-e-mail name list).
2. **My first leak metric was wrong** (distribution lists, common-name collisions): v1 20.8 % → 6.7 % after fixing it.
3. **LLM failure modes found by reading outputs** (v1→v2): wrong date arithmetic (now deterministic), French output for a French company, personal mail turned into work mail, renamed people, verbatim copying (10.4 → 3.1 %). Price: more retries (1.14 → 1.29 calls/e-mail) and forced redactions (0.8 → 5.8 %: real catches like `Kaminski` plus false positives on common words, later fixed).
4. **Infrastructure:** OpenRouter throttled >2 concurrent requests → global cap + backoff; then the account ran out of credit.

**Next:** judged v2; train-on-synthetic/test-on-real; cross-mailbox thread linking; stronger stage-1 de-id model.
