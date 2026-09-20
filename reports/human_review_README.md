# Human review protocol (30 pairs, ~30 minutes)

Open `human_review_sheet.csv`. For each row read `original_excerpt` and `synthetic_email`, then fill:

| Column | Scale | Question |
|---|---|---|
| realism_1to5 | 1-5 | Would you believe this is a real e-mail from a real company? (1 = obviously synthetic, 5 = indistinguishable) |
| original_identifier_left_YN | Y/N | Is any person / company / place / phone / address from the ORIGINAL still visible in the synthetic e-mail? |
| intent_preserved_YN | Y/N | Does the synthetic e-mail have the same purpose and tone (request, approval, complaint, personal chat ...)? |
| format_realistic_1to5 | 1-5 | Headers, greeting, sign-off, line wrapping, quoted history: do they look like real mail? |
| sender_signature_consistent_YN | Y/N | Is the person who signs the same as the sender in the header (when there is a signature)? |
| notes | text | Anything odd |

Then `python scripts/summarize_review.py`.
