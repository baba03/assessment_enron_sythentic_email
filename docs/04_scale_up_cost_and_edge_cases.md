# 4. Scale-up, cost, API handling and edge cases

## Cost (measured, not guessed)
Measured on 120 e-mails, including retries (~1.15-1.3 LLM calls per e-mail), from OpenRouter usage:

| Generator | USD / e-mail | 100 k e-mails | 1 M e-mails |
|---|---|---|---|
| Llama-3.3-70B (open-weight) | 0.00045 | ~US$45 | ~US$450 |
| Qwen3-235B | 0.00055 | ~US$55 | ~US$550 |
| GPT-4.1-mini (default) | 0.0012 | ~US$120 | ~US$1,200 |
| Gemini-2.5-Flash | 0.0017 | ~US$170 | ~US$1,700 |
| Claude Haiku 4.5 | 0.0043 | ~US$430 | ~US$4,300 |

Batch variants (`…:batch`) are listed by OpenRouter at about half price. The judge/evaluation pass costs more than generation (about 5 judge calls per e-mail), so evaluate a sample (e.g. 1-2 %), not everything. Only ~245 k distinct bodies exist in the corpus, so "1 M" would need multiple target companies per source e-mail (which the design supports).

## Throughput
Un-throttled, the 120-e-mail run took ~80 s (about 1.5 e-mails/s with 8 workers). On the assessment key, throughput was limited to 2 concurrent requests. For 1 M e-mails: async workers across machines, provider rate limits raised, and the response cache keyed by prompt (already implemented) to make re-runs free. Stage 1 costs milliseconds per e-mail and needs no API.

## What would change at scale
* Move stage 1 into a batch job (spaCy `nlp.pipe`), keep the LLM stage stateless and idempotent.
* Sample-based monitoring: run the independent leak checks + the LLM judge on 1-2 % of output; alert if the leak rate exceeds a threshold.
* Build the "known Enron people" list from the whole corpus (I used a 30 k-e-mail sample) for the verifier and for evaluation.
* Human review of a small stratified sample before releasing a dataset.
* Self-hosted open-weight generator (Llama/Qwen worked in the comparison) so confidential data never leaves the environment.

## API handling (implemented in `llm.py`)
Key from environment / `.env` (never committed; `.env.example` provided) · 60 s timeout · retries with backoff · separate wait-and-retry on 402/429 · global concurrency cap (`SYNTH_MAX_CONCURRENCY`) · per-process dollar cap (`max_cost`) · usage/cost accounting from OpenRouter · on-disk response cache · `SYNTH_OFFLINE=1` to forbid network calls · empty-completion retry. Output parsing is tag-based (`<subject>/<body>/<quoted>`) rather than JSON: it preserves whitespace and hard line breaks exactly, which matters for format realism. Malformed output → retry with feedback → best attempt kept.

## Edge cases: handled vs. not
| Case | Status |
|---|---|
| Empty subject | Handled (preserved, 5/5 in the sample) |
| `Re:` / `Fw:` prefixes | Handled (58/58 preserved) |
| Empty or very short body | Handled by the same path; sampling used 8-700 words |
| Very long body / quoted history | Truncated at 1,200 / 450 words, flagged in `meta` |
| Multiple recipients, Cc | Handled (headers rebuilt from the mapped people) |
| Forwarded / quoted history, Lotus & Outlook formats | Handled (rewritten to plain Outlook style) |
| Signatures, phone numbers, URLs, ids | Handled (Indian formats: +91 phones, rupee amounts via the LLM) |
| Dates in text, weekdays | Handled (whole-week shift) |
| Money / units | Delegated to the LLM (scaled to local currency); not automatically verified |
| Attachments | Only file-name mentions; no attachment content |
| HTML markup (0.5 % of e-mails), tables (8 %), bullets (7 %) | Pass through the LLM as text; **not in the 120-e-mail evaluation set**, so untested |
| Impossible dates (0.12 % of the corpus, 1979-2044) | Date shift falls back to a default; not separately tested |
| Invalid / refused / timed-out LLM output | Retry with feedback, backoff, best-attempt fallback, forced redaction |
| Names unusual, lowercase, misspelled | Partly handled (lexicon net); can still slip through |
| Non-English e-mails | Not specifically handled |
| Cross-mailbox threads | Not linked |
