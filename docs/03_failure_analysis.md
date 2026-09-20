# 3. Failure analysis: what broke, why, and what changed

Every row is a real defect found while reading outputs or from the independent evaluator; nothing here is hypothetical. "Fix" links to the code that now handles it.

| # | Failure observed | Root cause | Fix | Verified by |
|---|---|---|---|---|
| 1 | External person from a forwarded header survived (`"Vittal, Maheshram" <m…@wharton…>`) | spaCy misses `Last, First`; the address was replaced but the display name was not | Harvest `"Name" <addr>` pairs (`pii.py: PAIR_RE`); external people keep an external fake domain | Pilot re-run: replaced by a synthetic name |
| 2 | Famous surname (`Skilling`) survived in `To: Skilling, Jeff` | Header line without an address; not registered | Harvest names from `From:/To:/cc:` lines; always-blocked list of famous Enron surnames in the verifier | Re-run, no hit |
| 3 | In-text dates not shifted (`Tuesday, 16 Jan` next to a shifted header date) | LLMs do date arithmetic badly | Deterministic shift of numeric and textual dates by the same whole-week offset | Unit test `test_dates_in_text_shift_by_same_offset` |
| 4 | `Heather Kroll@ECT`, `IMCEANOTES-…` mailto blobs, `>@ENRON` in output | Lotus-Notes / Exchange artefacts | Strip them in stage 1 (`clean_artifacts`, `NOTES_SUFFIX`) | Manual audit + regression test |
| 5 | Wrapped text came back ragged (short orphan lines) | The LLM copied the original's 75-column line breaks, then my re-wrap broke them again | Unwrap before the LLM, re-wrap to the original width afterwards (`unwrap`/`rewrap`) | Manual audit |
| 6 | Output in **French** for the French company (earlier multi-country set) | Locale leaked into language | Prompt rule: language stays English; **and the default company set is now Indian-only**, so the issue cannot occur | 0 non-English greetings in the 120 v2 outputs (was 1) |
| 7 | A private, sensitive personal e-mail became a work e-mail (label drift) | "Domain transfer" over-applied | Prompt rule 0: type of communication is preserved | Manual audit; category κ effect to be re-measured (needs credit) |
| 8 | Real name `Alan Comnes` survived inside a Lotus recipient list | Names with double spaces, wrapped lines, `= =20` debris, or all-lowercase writers were invisible to the harvester | Whitespace-tolerant matching, quoted-printable repair, name-lexicon detector (capitalised, lowercase, wrapped) | Regression tests; all 4 leaking e-mails clean at stage 1 |
| 9 | Verifier flagged `time`, `outsourcing`, `memorandum`, then replaced them with random company names | NER tags common words as ORG | Word-frequency guard (`wordfreq` ≥ 3.3): still placeholder-replaced but not counted as a leak | Forced-redaction rate fell 18.3 % → 10.8 % (rule arm) |
| 10 | Signature name differed from the header sender (~7 % of e-mails with a sign-off) | LLM re-named a person that stage 1 had not registered | Name-preservation check plus new `signature_problem()` check: if the source is signed by the sender, the output must be too (retry with feedback) | Unit test; paired check 18/41 vs 19/41 in the originals; effect on real outputs to be re-measured (needs API) |
| 11 | My first leak metric reported 20.8 % for hybrid | Group labels ("the bank") and common-name collisions counted as Enron people | Given-name filter + per-e-mail synthetic-name exclusion | v1 20.8 % → 6.7 % |
| 12 | OpenRouter HTTP 402 `in_flight_budget_exhausted`, then account out of credit | >2 concurrent requests on a small key; then the shared balance ran out | Global concurrency cap, backoff, response cache, offline mode (`SYNTH_OFFLINE`) | Cache reproduces 80/120 outputs offline |
| 13 | Houston ZIP (`77002`) kept in the synthetic address | Quasi-identifier not covered | ZIP after `TX`/`Texas` replaced (`ZIP_RE`) | Unit test |
| 14 | `Manifesto 1-25-01.doc` kept the 2001 date | Only `m/d/y` dates were shifted | Dash-separated dates shifted too | Unit test |
| 15 | `x@dynegy.com@ENRON` → `…@cedarpointcap.com` | Exchange double-`@` artefact | Stripped in `clean_artifacts` | Unit test |
| 16 | A real-vs-synthetic classifier separated hybrid outputs from real mail (AUC 0.89) | LLM removes trailing spaces, uses curly quotes/dashes, single-space sentences | `restore_style()` reapplies source typography | AUC 0.89 → 0.69 on stored outputs (offline) |
| 17 | Names with middle initials / wraps in recipient lists survived in the LLM output and were invisible to my name-list check | Metric only matched `first last` | New independent survivor metric (own regex) | Final scrubber: 22 → 3 surviving names (rule arm) |
| 18 | Personal note (drill, house key) became a workplace vault-access note | LLM over-applies domain transfer | Prompt rule 0 (insufficient) | **Open** |

## Known unresolved issues (honest list)
* Judged v2 metrics (LLM realism score, label preservation) not measured: needs API credit. Offline substitutes: embedding retrieval and form-classifier AUC (docs/05).
* Personal-to-business drift (row 18) can still happen.
* 40/120 stored hybrid v2 outputs predate the last name-detection fixes.
* ~7 % of stored e-mails with a sign-off had a sender/signature mismatch (row 10); the new check has not yet been exercised on real LLM output.
* Generic and automated e-mails are only lightly re-worded; a small fraction still copies long phrases verbatim.
* No thread-level consistency across mailboxes; dates and companies are per mailbox.
