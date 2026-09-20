"""Builds notebooks/01_demo.ipynb and notebooks/02_analysis.ipynb (then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return n


demo = [
("md", """# Demo - Enron e-mail in, synthetic e-mail out

Runs the prototype end-to-end on real Enron e-mails.

**Pipeline** (`src/enron_synth/`): `parse` -> **stage 1: deterministic de-identification** (people, addresses, phones, ids, dates, Enron-isms) -> **stage 2: constrained LLM domain transfer** (OpenRouter) -> **verify + repair loop** -> render as a realistic raw e-mail.

**Offline / online.** Every LLM response is cached in `data/cache/`. With `OFFLINE = True` (default here) the notebook serves cached responses only and never touches the network, so it reproduces exactly and is free. Set `OFFLINE = False` (and put `OPENROUTER_API_KEY` in `.env`) to synthesise *new* e-mails - e.g. section 4."""),
("code", r"""import os, sys, warnings
OFFLINE = True
if OFFLINE:
    os.environ["SYNTH_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")
import pandas as pd
from enron_synth.pipeline import SyntheticEmailGenerator
from enron_synth.parsing import parse_email
from enron_synth.pii import Scrubber
from enron_synth.personas import COMPANY_BY_ID, PersonaFactory, COMPANIES
from enron_synth.llm import LLM, CacheMiss

gen = SyntheticEmailGenerator()          # default model: openai/gpt-4.1-mini via OpenRouter; default targets = 12 Indian companies
from enron_synth.personas import INTERNATIONAL_COMPANIES
gen_stored = SyntheticEmailGenerator(companies=INTERNATIONAL_COMPANIES)   # the stored LLM outputs (sections 2-3) were made with the earlier multi-country list
print("model:", gen.model, "| target companies:", [c.id for c in COMPANIES])

def synth(raw, file="", stored=False, **kw):
    try:
        return (gen_stored if stored else gen).generate(raw, file, **kw)
    except CacheMiss:
        print("(not in the response cache - set OFFLINE = False and provide an OpenRouter key to generate it)")"""),
("md", """## 1. The example e-mail from the assessment brief
Raw input, then **stage 1 only** - exactly what the LLM is allowed to see. People, addresses, phones, numbers and dates are already synthetic; organisations, places and the company name are `[PLACEHOLDERS]` that stage 2 must invent. The LLM never receives the original people."""),
("code", r"""raw = open("../tests/sample_email.txt").read()
print(raw)"""),
("code", r"""e = parse_email(raw, "fastow-a/sent/1.")
s = Scrubber(PersonaFactory(COMPANY_BY_ID["agriindia"])).scrub(e, shift_days=8400)
print("SUBJECT:", s.subject, "\n"); print(s.body); print("\nreplacement counts:", s.counts)"""),
("md", "Baseline **rule mode** (no LLM: placeholders filled from Faker) shows what de-identification alone gives - the business content is still Enron's:"),
("code", r"""print(gen.generate(raw, "fastow-a/sent/1.", company="kaveribank", mode="rule").raw)"""),
("md", "**Full hybrid pipeline** on the same e-mail (needs the API the first time; cached afterwards):"),
("code", r"""o = synth(raw, "fastow-a/sent/1.", company="agriindia")
if o: print(o.raw); print("\nmeta:", {k: o.meta[k] for k in ["model", "attempts", "leaks_after", "forced_redaction"]})"""),
("md", """**Recorded output** of that call from an earlier session (pipeline v1, `openai/gpt-4.1-mini`, target = Agriculture India). The Enron text is the brief's example; the output below was produced by the pipeline, not written by hand:

```
Message-ID: <caee4903d563ef0c.58556@agriindia.com>
Date: Thu, 06 Jun 2024 07:48:00 +0530
From: Advik Kara <advik.kara@agriindia.com>
To: Balhaar Sanghvi <balhaar.sanghvi@agriindia.com>
Subject: Approval of the DPR Kaveri Commodities transaction
Cc: Gabriel Nazareth <gabriel.nazareth@agriindia.com>
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii

Dear Balhaar,

Attached is the DASH for the approval of the DPR Kaveri Commodities transaction. This partial divestiture allows us to return ₹8.2 crore of our equity interest back to Aryan Grains, Pvt. Ltd., and its subsidiary, Pune Agro Holdings. Both entities are controlled by Charles Dasgupta.

In addition to redeeming part of our equity interest, the deal provides us with 900,000 quintals of wheat priced below market, an option which could lead to a very profitable flour milling project, and the potential for more brokerage fees from other Dasgupta-affiliated firms.

The DASH has been approved and signed by RAC and AgriIndia's Farm Partnerships unit and is now awaiting Gabriel Nazareth's review and approval. I wanted to give you the opportunity to review the DASH and become familiar with the provisions of the deal.

If you have any questions on the transaction, feel free to contact me at +91 98515 42387. Others familiar with the deal are Advaith Tiwari, Tanish Bumb, and Christopher Chauhan.

Thank you.

Best regards,
Advik Kara
Agriculture India Pvt. Ltd., Pune
```

Note what v1 still got wrong (and v2 fixes by instructing the LLM to replace internal jargon): the Enron-internal terms **DASH** and **RAC** survived. Everything identifying (people, company, counterparties, amounts, phone, city) was replaced consistently, in the domain of the target company."""),
("md", """## 2. Random e-mails from the evaluation set (stored LLM outputs from the earlier multi-country run)
Original vs. synthetic: hard-wrapping, forwarded history, tone and structure are preserved; people, companies, places, numbers, dates and currency are not."""),
("code", r"""ev = pd.read_json("../data/eval_set.jsonl", lines=True)
def show(i, n=1300):
    r = ev.iloc[i]
    body = r["message"].split("X-FileName:")[1].split("\n", 1)[1]
    o = synth(r["message"], r["file"], stored=True, seed=str(i))
    print("#" * 100, f"\nORIGINAL  ({r['file']})\n" + body[:n], "\n\n----------------- SYNTHETIC -----------------\n")
    if o:
        print(o.raw[:n + 350]); print("\nmeta:", {"company": o.company} | {k: o.meta[k] for k in ["attempts", "leaks_after", "forced_redaction"]})
for i in [3, 11, 20]:
    show(i)"""),
("md", """## 3. Consistency
Same Enron person -> same synthetic person inside a mailbox (threads stay coherent); different mailbox -> different company. Names below are the synthetic senders for one Enron mailbox:"""),
("code", r"""rows = []
for i in range(len(ev)):
    r = ev.iloc[i]
    if r["file"].startswith("kaminski-v"):
        try:
            o = gen_stored.generate(r["message"], r["file"], seed=str(i))
        except CacheMiss:
            continue
        rows.append((o.company, o.headers["From"], o.headers["To"][:60], o.headers["Date"]))
pd.DataFrame(rows, columns=["company", "From", "To", "Date"]).head(8)"""),
("md", """## 4. Your own e-mail
Paste any raw Enron message (headers + body). Needs `OFFLINE = False` + an API key the first time. CLI equivalent: `python -m enron_synth.cli one --input my_email.txt`"""),
("code", r"""my_email = '''Message-ID: <1.2.JavaMail.evans@thyme>
Date: Tue, 3 Oct 2000 09:12:00 -0700 (PDT)
From: sara.shackleton@enron.com
To: mark.taylor@enron.com
Subject: Re: ISDA master - Credit Suisse
X-From: Sara Shackleton
X-To: Mark E Taylor

Mark,

Please find attached the revised ISDA schedule for Credit Suisse First Boston. I have flagged the two provisions that the credit group wants to negotiate. This is privileged and confidential - do not forward outside legal.

Let's talk Thursday morning.

Sara
'''
o = synth(my_email, "shackleton-s/sent/1.")
if o: print(o.raw)
print(LLM.report())"""),
]

analysis = [
("md", """# Analysis - EDA and experiment results

Everything here is read from artifacts produced by `python -m enron_synth.eda` and `python -m enron_synth.experiment` (no API calls)."""),
("code", r"""import json, pandas as pd, warnings
warnings.filterwarnings("ignore")
from IPython.display import Image, Markdown, display
eda = json.load(open("../reports/eda_summary.json")); dup = json.load(open("../reports/dup_stats.json"))
pd.Series({k: v for k, v in eda.items() if not isinstance(v, dict)}).to_frame("value")"""),
("code", r"""display(Image("../reports/figures/eda_overview.png"))
print("Duplicates:", {k: v for k, v in dup.items() if k != "duplicate_rows_by_folder_top10"})"""),
("md", "## Results (n = 120 stratified e-mails)"),
("code", r"""display(Markdown("### Privacy & fidelity - every arm, n = 120 (offline metrics)"))
display(Markdown(open("../reports/results_offline.md").read()))
display(Image("../reports/figures/results_comparison.png"))
display(Markdown("### LLM-judged realism & label preservation (v1 pipeline arms, judge = Claude Haiku 4.5)"))
display(Markdown(open("../reports/results_judged.md").read()))
display(Image("../reports/figures/results_judged.png"))"""),
("md", "## Additional offline evaluation (no API): embeddings, real-vs-synthetic classifier, name survivors, consistency, diversity"),
("code", r"""display(Markdown(open("../reports/results_extra.md").read()))
extra = json.load(open("../reports/eda_extra.json"))
print({k: extra[k] for k in ["unique_senders", "unique_recipients", "unique_sender_domains", "pct_internal_only", "unique_normalised_subjects", "pct_subjects_with_>1_message"]})
display(Image("../reports/figures/eda_extra.png"))"""),
("md", "## Label preservation per field (hybrid v1, judged)"),
("code", r"""res = json.load(open("../reports/results_judged_v1.json"))
pd.DataFrame(res["hybrid"]["utility"]).T.round(3)"""),
]

(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb(demo), ROOT / "notebooks" / "01_demo.ipynb")
nbf.write(nb(analysis), ROOT / "notebooks" / "02_analysis.ipynb")
print("ok")
