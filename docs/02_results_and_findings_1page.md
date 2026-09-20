# 2. Experiment results and findings (one-page version)

> **Status:** LLM-judged results exist only for pipeline v1 (the OpenRouter account ran out of credit mid-evaluation). v2 is verified by offline metrics, tests and manual audit; `scripts/finish_evaluation.sh` completes the rest (~US$2).

**Data analysis (517,401 e-mails).** Mail is short and informal: median body 45 words, 8.6 % empty, greeting in 5 %, sign-off in 18 %. 36 % carry quoted/forwarded history. 48 % mention "Enron", 31 % embed e-mail addresses, 24 % phone numbers. Enron-isms betray the source without names (`/HOU/ECT@ECT`, `EOL`, 75-column wrap, quoted-printable debris). 52.7 % of rows are body-duplicates, so dedupe before sampling.

**Synthesis process.** Stage 1 (deterministic, one regex pass) replaces header identities (nickname-aware), `"Last, First" <addr>` pairs, spaCy persons, a given-name-lexicon net, addresses, phones, ids and Enron-isms; orgs/places/company become placeholders. Persons are a keyed hash, so the same Enron person is the same synthetic person across a mailbox. Each mailbox maps to one of 12 fictional Indian companies (Indian names, rupees, +91 phones, English); dates shift by whole weeks (weekday, clock time, gaps preserved). Stage 2: an LLM transfers the business domain, seeing only pseudonymised text. Then verify/repair (retry ≤3, forced redaction).

| Arm (n=120) | Any leak ↓ | 8-gram copy ↓ | Length ρ ↑ | Structure ↑ | Judge spots synthetic ↓ | Category κ ↑ |
|---|---|---|---|---|---|---|
| Naive LLM only | 55.8 % | 5.3 % | 0.41 | 73.5 % | 83 % | 0.80 |
| Rule-based only | 11.7 % | 52 % | 1.00 | 99.8 % | 69 % | 0.82 |
| Hybrid v1 | 6.7 % | 10.4 % | 0.99 | 99.0 % | 63 % | 0.69 |
| Hybrid v2 | 3.3 % (4.2 % strict) | 3.1 % | 0.99 | 99.4 % | n/a (form classifier: AUC 0.69) | n/a (own-source retrieval 89 %) |

Offline checks on v2 (`docs/05`): embedding retrieval finds each synthetic e-mail's own source 89 % of the time (chance <1 %); a form-only real-vs-synthetic classifier detected v2 at AUC 0.89 until I found and restored the LLM's typography habits (trailing spaces, curly quotes, sentence spacing), giving 0.69. Hybrid beats the naive LLM clearly (leak 95 % CIs: 3-13 % vs 47-64 %); v1-vs-v2 and hybrid-vs-rule are within CI. Realism is not solved (judge picks the synthetic e-mail 63 % of the time; chance = 50 %). Intent 87 %, privilege 96 %, sensitive 89 % agreement; category κ fell because v1 sometimes turned personal mail into work mail. Llama-3.3-70B (open-weight) leaks 3.3 % at US$0.0004/e-mail.

**Challenges.** (1) The independent evaluator found real leaks my verifier missed (names in Lotus-Notes lists with double spaces, wraps, `= =20` debris, all-lower-case writers); fixed and regression-tested. 40/120 hybrid outputs need regeneration with the final scrubber, so 3.3 % is a lower bound. (2) My first leak metric was wrong (group labels, common-name collisions): 20.8 % → 6.7 % after fixing. (3) LLM failures found by reading outputs: date arithmetic (now deterministic), French output, personal mail turned into work mail, verbatim copying (10.4 → 3.1 %). (4) OpenRouter throttled >2 concurrent requests (global cap added), then the account ran out of credit.
