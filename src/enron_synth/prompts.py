"""Prompt templates. Kept in one file so they can be versioned / ablated easily."""

SYSTEM_HYBRID = """You are a synthetic-data engine used by an eDiscovery research team. You convert a real corporate e-mail (already partly de-identified) into a NEW e-mail that looks like it was written by an employee of a DIFFERENT, fictional company. The result is used to fine-tune and evaluate language models, so it must be indistinguishable in form and style from genuine business e-mail - never a summary, paraphrase-with-commentary or assistant-style answer.

Output ONLY the tagged blocks requested. No preamble, no notes."""

USER_HYBRID = """## TARGET COMPANY (all events now happen here)
- Name: {company_name}   (employees say "{company_short}")
- Industry: {industry}
- Location: {city}, {country}   Money: {currency_note}
- Business units you may reference: {units}
- E-mail domain: {domain}
- Sender (write the signature for THIS person if the original has a signature): {sender_name}, {sender_title}. If a signature needs a phone number / address that is not in the input, invent one in {country}'s format; if the input already contains the sender's number, reuse it.

## WHAT TO PRESERVE (this is what downstream classifiers learn from)
0. TYPE OF COMMUNICATION: a personal / social / family / gossip / sports / jokes / health / job-hunting message stays personal (it is NOT turned into a work message); a legal, HR, finance, IT, scheduling or automated / newsletter message keeps that type. Only the concrete details change. Write in ENGLISH regardless of the target country (names and places are local, the language is English) unless the original is in another language.
1. Communicative intent, tone, formality, sentiment, urgency and the roles of the participants (who asks / approves / complains / advises whom).
2. Structure & surface form: same number of paragraphs, lists, greeting / sign-off presence and style, capitalisation habits (e.g. all-lowercase, ALL CAPS), abbreviations, typos and terse style. KEEP LINE BREAKS: every signature / address / list line stays on its own line exactly as in the input. {wrap_hint}
3. Wording: re-word sentences that carry concrete details (do not reproduce runs of 8+ identical words from the input), but keep boilerplate, greetings, sign-offs and the writer's voice. Length: about {n_words} words in the new body (allowed range {lo}-{hi}).
4. Legal / sensitive cues: if the original signals privilege, confidentiality, litigation, HR, compliance, a regulator, a deal or a threat, keep that signal (invent new details) - do NOT sanitise or soften it.
5. Calls to action, questions, deadlines and relative time references (e.g. "by Friday", "next week").

## WHAT TO CHANGE
1. DOMAIN TRANSFER: rewrite business context - commodities, counterparties, deals, projects, systems, products, jargon - into believable equivalents for {industry}. Do it consistently across the whole e-mail. If the e-mail is generic or personal (lunch, gossip, jokes), keep it generic but change every concrete detail.
2. Placeholders in [SQUARE BRACKETS] were inserted by an anonymiser. Replace EVERY one by an invented, plausible value for the target company / locale, consistently (an invented organisation must be a NEW plausible name - never the target company's own name, and never a city name) ({placeholder_list}). [COMPANY] = the target company. No square-bracket placeholder may remain in your output.
3. People's names, e-mail addresses and phone numbers in the input are ALREADY synthetic: keep them exactly. Company-internal jargon, acronyms, form / system / committee names (e.g. all-caps internal terms such as DASH, RAC, EOL-style names) are identifiers of the old company: replace them with plausible internal equivalents of the target company. If you notice any other real-looking person, organisation, place, product, project, deal or building name that survived (for example the old company "Enron", "Houston", energy-market entities), replace it with an invented one that fits the target world.
4. Numbers: change amounts, quantities, prices and percentages to plausible ones for the target industry and currency, but keep relationships (ratios, "doubled", "10% higher", totals that add up). Rename attachments / file names.
5. Dates: this e-mail is now dated {new_date}. Calendar dates in the input have ALREADY been shifted consistently (weekdays match) - keep those dates and their day/month exactly; you may only re-format numeric dates to the convention of {country}. Adapt any remaining time-specific facts (years, quarters, events, "last year") so they are plausible for {new_year}.
6. Quoted / forwarded material: keep the same structure and nesting, but rewrite its header lines in a plain Outlook style ("-----Original Message-----", From:, Sent:, To:, Subject:) using the target-company people and addresses that appear in the input. Remove Lotus-Notes artefacts (e.g. "/HOU/ECT@ECT").

## INPUT (subject, then body, then quoted material)
<subject>{subject}</subject>
<body>
{body}
</body>
<quoted>
{trailer}
</quoted>
{feedback}
## OUTPUT FORMAT
Return exactly:
<subject>...</subject>
<body>
...
</body>
<quoted>
...   (leave empty if the input quoted block was empty)
</quoted>"""

FEEDBACK_TMPL = """
## FIX REQUIRED (your previous attempt was rejected)
{problems}
Rewrite the e-mail again and fix these problems.
"""

SYSTEM_LLM_ONLY = "You rewrite emails."
USER_LLM_ONLY = """Rewrite the following email so that it appears to come from {company_name} ({industry}, {country}). Change all names, companies, places, phone numbers, email addresses and other identifying details so the original cannot be recognised, but keep the meaning and style.

Return exactly:
<subject>...</subject>
<body>
...
</body>

EMAIL:
From: {from_}
To: {to}
Subject: {subject}

{body}
{trailer}"""

# ------------------------------------------------------------------ evaluation prompts
JUDGE_SYSTEM = "You are a meticulous evaluator of e-mail datasets for enterprise LLM fine-tuning. Reply with JSON only."

JUDGE_REALISM = """Below is an e-mail (headers + body). Decide how much it looks like a GENUINE e-mail written by an employee of a real company, judged on FORM only (headers, greeting, sign-off, formatting, register, internal consistency of names / dates / numbers, natural imperfections). Do not judge whether the content is interesting.

Score 1-5: 1 = obviously machine-written / inconsistent, 3 = plausible but slightly polished or generic, 5 = indistinguishable from real business e-mail.
Also list any specific artefacts (max 3) such as: over-polished tone, wrong name reuse, placeholder remnants, unrealistic numbers, assistant-style phrasing, inconsistent dates, mismatch between header names and body names.

EMAIL:
{email}

Return JSON: {{"realism": <1-5>, "artifacts": ["..."]}}"""

JUDGE_LABELS = """Classify this e-mail for an eDiscovery review workflow. Reply with JSON only.

Fields:
- "category": one of ["deal_transaction","scheduling_logistics","legal_compliance","hr_personnel","finance_accounting","technical_it","operations_trading","personal_social","newsletter_automated","other"]
- "intent": one of ["request_action","provide_information","approval_decision","complaint_dispute","scheduling","social_chitchat","legal_advice","other"]
- "privileged_likely": true/false  (looks like legal advice / attorney-client communication)
- "sentiment": one of ["negative","neutral","positive"]
- "action_required": true/false
- "sensitive": true/false (contains personal, financial-distress, litigation, regulatory or confidential-deal information)

EMAIL:
{email}

Return JSON: {{"category":"...","intent":"...","privileged_likely":false,"sentiment":"...","action_required":false,"sensitive":false}}"""

JUDGE_PAIRWISE = """Two e-mails follow. One is a REAL corporate e-mail; the other is a SYNTHETIC e-mail generated from a different real e-mail. Judge only the form and naturalness of the writing, headers and formatting. Which one is the real e-mail?

=== EMAIL A ===
{a}

=== EMAIL B ===
{b}

Return JSON: {{"real": "A" or "B", "confidence": <0.5-1.0>, "reason": "<one sentence>"}}"""
