# Next steps: the 4 open items

## 1. Get OpenRouter credit, then finish the evaluation

**Why:** the OpenRouter *account* balance is exhausted (`GET /api/v1/credits` → `total_credits: 390`, `total_usage: 390.19`). The key's own US$40 limit is not the cause, and this project used only ~US$3.2. Requests with prompts above ~1.3k tokens are refused with HTTP 402, so LLM-judge scoring of the final pipeline and the model comparison could not run.

**Check the balance (should show usage < credits):**
```bash
curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

**Message to send to whoever issued the key:**
> Hi, the OpenRouter key you gave me for the Consilio technical assessment stopped working mid-evaluation. The account-level balance is exhausted (`/api/v1/credits` reports 390.19 used of 390), even though the key's own limit shows ~US$37 remaining, so requests fail with HTTP 402 "Prompt tokens limit exceeded". My own usage was about US$3.2. Could you top up the account (about US$2-3 is enough) or send a replacement key? Everything else is finished, and one script completes the remaining LLM-judged results. Thanks!

**Once credit is back:**
```bash
cd enron-synthetic-emails
pip install -e .
sh scripts/finish_evaluation.sh      # ~US$2-3: regenerates the hybrid arm with the Indian company set (all 120 e-mails, ~US$0.15), judges it, judges 4 alternative models
```
Then update the "Status" block in `README.md` and the `Hybrid v2` row of `docs/02_results_and_findings.md` (realism / category κ columns) with the new numbers in `reports/results_judged_final.json`.

## 2. Publish to GitHub

Nothing has been committed or pushed. `.env` (your API key) is git-ignored; keep it that way.

```bash
cd enron-synthetic-emails
git init && git add . && git status          # confirm .env is NOT listed
git commit -m "Synthetic Enron e-mail generation prototype"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
`data/cache/` (9 MB) is included on purpose so results reproduce offline (`SYNTH_OFFLINE=1`). Then email the repo link as the brief requires. Because the API key was pasted in chat, consider rotating it once the assessment is over.

## 3. Verify the references (they are from memory, not a live search)

`docs/01_experiment_plan.md` cites the following. Please check each exists and says what the plan claims before submitting:

| Cited as | Full reference to look up | Claim it supports |
|---|---|---|
| Carrell et al. 2013 | Carrell D. et al., "Hiding in plain sight: use of realistic surrogates to reduce exposure of protected health information in clinical text", *JAMIA* 2013 | realistic surrogates beat `[NAME]` redaction |
| Carlini et al. 2021 | Carlini N. et al., "Extracting Training Data from Large Language Models", USENIX Security 2021 | LLMs memorise training data |
| The Pile | Gao L. et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", 2020 (contains an *Enron Emails* subset) | Enron is in LLM pre-training data |
| Zheng et al. 2023 | Zheng L. et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023 | LLM-as-judge, self-preference bias |
| Presidio, Faker | github.com/microsoft/presidio, github.com/joke2k/faker | PII detection / surrogate generation tooling |

If any cannot be verified, delete that citation (the argument in the plan does not depend on any single one).

## 4. Length of the results document

The brief caps deliverable 2 at **1 page**. `docs/02_results_and_findings.md` is the fuller version (~700 words). A strict one-page version is in `docs/02_results_and_findings_1page.md`; submit whichever you prefer (the fuller one is better for reviewers who want the evidence, the short one for the page limit).

## 5. (New) Optional: a human reviewer
`reports/human_review_sheet.csv` holds 30 original/synthetic pairs; the protocol is in `reports/human_review_README.md`. Fill it (~30 min), then `python scripts/summarize_review.py`. My own audit is in `docs/05_additional_evaluation_and_audit.md`; it is not a substitute for a human review.
