"""Stage 1 - deterministic de-identification.

Everything that can be replaced *reliably* is replaced here, in ONE regex pass (so a freshly inserted
synthetic name can never be re-matched and re-replaced):

  people    header-derived identities (From/To/Cc + X-From/X-To) and spaCy PERSON entities -> synthetic people
  contact   e-mail addresses, phone numbers, URLs, SSN-like and long numeric ids            -> synthetic values
  Enron     the company name, unit acronyms, known Enron-world orgs / places, Lotus-Notes "/HOU/ECT" suffixes
            and spaCy ORG / GPE / LOC entities                                             -> [COMPANY], [UNIT_n], [ORG_n], [LOC_n]

The bracketed placeholders are handed to the LLM (stage 2), which fills them with *industry- and locale-
appropriate* inventions. Because the LLM only ever sees this scrubbed text, it does not need to see
the original people at all (a useful property when the real data is confidential).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache

from wordfreq import zipf_frequency

from .enron_terms import (COMMON_WORD_SURNAMES, COMPANY_TERMS, FAMOUS_SURNAMES, LOC_TERMS, NER_STOPLIST, NICKNAMES, ORG_TERMS,
                          UNIT_ACRONYMS)
from .parsing import ParsedEmail
from .personas import Company, Person, PersonaFactory, _h, guess_gender

FREEMAIL = {"gmail.com", "yahoo.com", "hotmail.com", "aol.com", "msn.com", "earthlink.net", "comcast.net", "att.net",
            "sbcglobal.net", "mindspring.com", "outlook.com", "juno.com", "netzero.net", "excite.com", "home.com"}

EMAIL_RE = r"[A-Za-z0-9._%+'\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"
URL_RE = r"(?:https?://|www\.)[^\s<>\"')\]]+"
SSN_RE = r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
PHONE_RE = r"(?:(?<!\d)\+?1[\s.\-])?(?<!\d)\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)(?:\s?(?:x|ext\.?)\s?\d{1,5})?"
LONGNUM_RE = r"(?<![\d.,])\d{7,}(?![\d.,])"
NOTES_SUFFIX = r"(?:(?:/[A-Za-z]{2,8}){1,3}(?:@[A-Za-z]{2,12})?|@(?:ECT|ENRON|EES|ENA|EWS|EGM|EBS|EI|EIM|ENRONXGATE|Enron|enronXgate|ees))"

MONTHS = "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
NDATE_RE = r"(?<![\d/.-])(\d{1,2})(?P<sep>[/-])(\d{1,2})(?P=sep)(\d{4}|\d{2})(?![\d/-])"
TDATE_RE = rf"\b({MONTHS})\.?\s+(\d{{1,2}})(st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b"
TDATE2_RE = rf"\b(\d{{1,2}})(st|nd|rd|th)?\s+({MONTHS})\b(?:,?\s+(\d{{4}}))?"
_MON = {m[:3].lower(): i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
PAIR_RE = re.compile(r'"([^"<>\n@]{3,50})"\s*<\s*([A-Za-z0-9._%+\'\-]+@[A-Za-z0-9\-.]+)\s*>|(?:(?<=: )|(?<=, ))([A-Z][A-Za-z.\'\-]+(?: [A-Z][A-Za-z.\'\-]+){1,3})\s*<\s*([A-Za-z0-9._%+\'\-]+@[A-Za-z0-9\-.]+)\s*>')
ZIP_RE = r"(?:(?<=TX )|(?<=TX, )|(?<=TX  )|(?<=Texas )|(?<=Texas, )|(?<=Tex\. ))\d{5}(?:-\d{4})?(?!\d)"
OFFICE_RE = r"\bEB\s?\d{3,4}[A-Za-z]?\b"


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler", "tagger"])
    except OSError as e:  # pragma: no cover
        raise RuntimeError("Run: python -m spacy download en_core_web_sm") from e


# ----------------------------------------------------------------------------- name helpers
def _title(tok: str) -> str:
    return tok if not tok.isupper() or len(tok) <= 2 else tok.capitalize()


def clean_display_name(s: str) -> str:
    """'Tim Belden <Tim Belden/Enron@EnronXGate>' -> 'Tim Belden' ; 'Allen, Phillip K.' -> 'Phillip K Allen'."""
    s = re.sub(r"<[^>]*>", "", s or "")
    s = re.sub(NOTES_SUFFIX, "", s)
    s = s.strip(" \"'\t\n")
    if "@" in s:
        s = s.split("@")[0]
    if s.count(",") == 1:
        last, first = [x.strip() for x in s.split(",")]
        if last and first and len(last.split()) <= 2 and len(first.split()) <= 3:
            s = f"{first} {last}"
    s = re.sub(r"\s+", " ", s.replace(".", " ") if re.fullmatch(r"[A-Za-z.\- ]+", s) else s).strip()
    return " ".join(_title(t) for t in s.split())


def name_from_address(addr: str) -> str:
    local = addr.split("@")[0]
    parts = [p for p in re.split(r"[._\-]", local) if p]
    if len(parts) >= 2 and all(p.isalpha() and len(p) > 1 for p in parts[:2]):
        return " ".join(p.capitalize() for p in parts)
    return local  # opaque id like 'pallen'


def _first_last(name: str) -> tuple[str, str, list[str]]:
    toks = [t for t in re.sub(r"[.,]", " ", name).split() if t.lower() not in {"jr", "sr", "ii", "iii", "iv", "phd", "md"}]
    if not toks:
        return "", "", []
    if len(toks) == 1:
        return toks[0], "", []
    return toks[0], toks[-1], toks[1:-1]


# ----------------------------------------------------------------------------- result container
@dataclass
class Scrubbed:
    company: Company
    subject: str
    body: str
    trailer: str
    from_person: Person
    to_people: list[Person]
    cc_people: list[Person]
    placeholders: dict[str, str]                     # '[ORG_1]' -> kind (for the prompt)
    hard_terms: set[str] = field(default_factory=set)   # must never appear in the final output
    soft_terms: set[str] = field(default_factory=set)   # first names etc. (warn only)
    counts: dict[str, int] = field(default_factory=dict)
    original_words: set[str] = field(default_factory=set)
    synthetic_names: set[str] = field(default_factory=set)   # every synthetic person created for this e-mail
    keep_names: list[str] = field(default_factory=list)      # synthetic names that occur in the scrubbed text (LLM must keep them)


# ----------------------------------------------------------------------------- scrubber
class Scrubber:
    def __init__(self, factory: PersonaFactory, use_ner: bool = True):
        self.f = factory
        self.use_ner = use_ner

    # ---- identity building -------------------------------------------------------------
    def _register_person(self, name: str, aliases: dict[str, str], hard: set, soft: set, gender_hint=None) -> Person | None:
        name = clean_display_name(name)
        first, last, mid = _first_last(name)
        if not first or len(first) < 2:
            return None
        f_low = first.lower()
        formal = NICKNAMES.get(f_low, f_low)
        key_name = f"{formal.capitalize()} {last}".strip()      # 'Bill Giuliani' and 'William Giuliani' are ONE person
        p = self.f.person_for(key_name, gender_hint)
        variants = {first}
        if last or True:
            variants |= {formal.capitalize()} | {n.capitalize() for n, fm in NICKNAMES.items() if fm == formal}
        cands: list[tuple[str, str]] = []
        for fv in sorted(variants):
            if last:
                forms = {f"{fv} {last}", f"{fv} {' '.join(mid)} {last}".replace("  ", " ")}
                for m in mid:
                    forms.add(f"{fv} {m}. {last}")
                for ff in forms:
                    cands.append((ff, p.full))
                for lf in (f"{last}, {fv}", f"{last},{fv}") + tuple(f"{last}, {fv} {m}." for m in mid) + tuple(f"{last}, {fv} {m}" for m in mid):
                    cands.append((lf, f"{p.last}, {p.first}"))
                hard.add(f"{fv} {last}".lower())
            if len(fv) >= 3:
                cands.append((fv, p.first))
                soft.add(fv.lower())
        if last and len(last) >= 4 and last.lower() not in COMMON_WORD_SURNAMES:
            cands.append((last, p.last))
            hard.add(last.lower())
        for orig, repl in cands:
            if orig and orig not in aliases:
                aliases[orig] = repl
                if orig.upper() != orig and len(orig) > 2:
                    aliases.setdefault(orig.upper(), repl.upper())
                # all-lowercase writers: full names and surnames (never bare first names - too many common words)
                if (" " in orig or (last and orig == last)) and "," not in orig:
                    aliases.setdefault(orig.lower(), repl.lower())
        return p

    # ---- main entry --------------------------------------------------------------------
    def scrub(self, e: ParsedEmail, max_body_words: int = 1200, max_trailer_words: int = 450,
              shift_days: int = 0) -> Scrubbed:
        aliases: dict[str, str] = {}
        hard: set[str] = set()
        soft: set[str] = set()
        counts: dict[str, int] = {"person": 0, "email": 0, "phone": 0, "url": 0, "id": 0, "org": 0, "loc": 0, "unit": 0, "company": 0}

        # 1) header identities
        def names_for(addrs: list[str], xline: str) -> list[str]:
            frags = [clean_display_name(x) for x in re.split(r",\s*(?![^<]*>)", xline)] if xline else []
            if frags and len(frags) == len(addrs) and all(frags):
                return frags
            return [name_from_address(a) for a in addrs]

        from_name = clean_display_name(e.x_from) or name_from_address(e.from_addr) if e.from_addr else ""
        to_names = names_for(e.to_addrs, e.x_to)
        cc_names = [name_from_address(a) for a in e.cc_addrs]
        addr_owner: dict[str, Person] = {}
        from_p = self._register_person(from_name or "Unknown Sender", aliases, hard, soft)
        if e.from_addr and from_p:
            addr_owner[e.from_addr.lower()] = from_p
        to_ps, cc_ps = [], []
        for a, n in zip(e.to_addrs, to_names):
            p = self._register_person(n, aliases, hard, soft)
            if p:
                addr_owner[a.lower()] = p
                to_ps.append(p)
        for a, n in zip(e.cc_addrs, cc_names):
            p = self._register_person(n, aliases, hard, soft)
            if p:
                addr_owner[a.lower()] = p
                cc_ps.append(p)
        counts["person"] = len(to_ps) + len(cc_ps) + 1

        # trim over-long input (rare newsletters / reports); cut on a line boundary
        body = clean_artifacts(_trim_words(e.body, max_body_words))
        trailer = clean_artifacts(_trim_words(e.trailer, max_trailer_words))
        subject = e.subject
        original_text = "\n".join([subject, body, trailer])

        # 1b) "Display Name" <addr> pairs anywhere in the text (quoted headers of forwarded mail): NER is poor at
        #     "Last, First" forms, so harvest them explicitly. External people keep an *external* fake domain.
        ext_email: dict[str, str] = {}
        for m in PAIR_RE.finditer(_ws(original_text)):
            nm, ad = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
            nm = (nm or "").strip(" \"'")
            if not nm or "@" in nm or any(c.isdigit() for c in nm) or len(nm.split()) > 4 or nm.lower() in NER_STOPLIST:
                continue
            p = self._register_person(nm, aliases, hard, soft)
            adl = ad.lower()
            if p and adl not in addr_owner:
                dom = adl.split("@", 1)[1]
                if dom == "enron.com" or dom.endswith(".enron.com"):
                    addr_owner[adl] = p
                else:
                    ext_email[adl] = f"{self.f._ascii(p.first).lower()}.{self.f._ascii(p.last).lower()}@{_fake_domain(self.f, dom)}".replace(" ", "")
                    hard.add(dom)

        for nm in harvest_header_names(_ws(original_text)):
            if len(nm.split()) <= 4:
                self._register_person(nm, aliases, hard, soft)
        for nm in harvest_capitalized_names(_ws_join(original_text)) + harvest_lowercase_names(_ws_join(original_text)):
            self._register_person(nm, aliases, hard, soft)
        for sn in FAMOUS_SURNAMES:
            hard.add(sn.lower())

        # 2) NER on original text -> more people / orgs / locations
        placeholders: dict[str, str] = {}
        ent_map: dict[str, str] = {}          # exact string -> placeholder (case-sensitive)
        ci_map: dict[str, str] = {}           # lower-case string -> placeholder (case-insensitive, blocklists)
        acr_map: dict[str, str] = {}
        n_org = n_loc = n_unit = 0

        def ph(kind: str, n: int) -> str:
            tag = f"[{kind}_{n}]"
            placeholders[tag] = kind
            return tag

        placeholders["[COMPANY]"] = "COMPANY"
        for t in COMPANY_TERMS:
            ci_map[t.lower()] = "[COMPANY]"
            hard.add(t.lower())
        hard.add("enron")
        hard.add("houston")
        # blocklist orgs/locs/acronyms are numbered lazily on first use inside the text
        used_lazy: dict[str, str] = {}

        if self.use_ner and original_text.strip():
            doc = _nlp()(original_text[:20000])
            for ent in doc.ents:
                txt = ent.text.strip().strip("'\"()[]<>,.;:")
                if len(txt) < 3 or "\n" in txt or txt.lower() in NER_STOPLIST or "@" in txt or txt.isdigit():
                    continue
                if not any(ch.isupper() for ch in txt):
                    continue
                if ent.label_ == "PERSON":
                    txt = re.sub(r"'s$", "", txt)
                    first, last, _ = _first_last(clean_display_name(txt))
                    if not first or any(w.lower() in NER_STOPLIST for w in txt.split()) or any(c.isdigit() for c in txt):
                        continue
                    if txt in aliases or (len(txt.split()) == 1 and txt.capitalize() in aliases):
                        continue
                    if len(txt.split()) == 1 and guess_gender(txt) == "u":
                        continue   # single word that is not a known first name: NER false positive (e.g. 'Trip')
                    self._register_person(txt, aliases, hard, soft)
                    counts["person"] += 1
                elif ent.label_ == "ORG" and txt not in ent_map:
                    if txt.isupper() and len(txt) <= 4 and txt not in UNIT_ACRONYMS:
                        continue
                    if txt.lower() in ci_map:
                        continue
                    n_org += 1
                    ent_map[txt] = ph("ORG", n_org)
                    if not _too_common_to_block(txt, original_text):
                        hard.add(txt.lower())
                    counts["org"] += 1
                elif ent.label_ in ("GPE", "LOC", "FAC") and txt not in ent_map:
                    n_loc += 1
                    ent_map[txt] = ph("LOC", n_loc)
                    if not _too_common_to_block(txt, original_text):
                        hard.add(txt.lower())
                    counts["loc"] += 1

        # numbering for blocklist items that actually occur
        lowered = original_text.lower()
        for t in ORG_TERMS:
            if re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", lowered) and t.lower() not in ci_map:
                n_org += 1
                ci_map[t.lower()] = ph("ORG", n_org)
                hard.add(t.lower())
        for t in LOC_TERMS:
            if re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", lowered) and t.lower() not in ci_map:
                n_loc += 1
                ci_map[t.lower()] = ph("LOC", n_loc)
                hard.add(t.lower())
        for t in UNIT_ACRONYMS:
            if re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", original_text):
                n_unit += 1
                acr_map[t] = ph("UNIT", n_unit)
                hard.add(t.lower())

        # 3) one master regex, one pass
        rng_key = self.f.company.id

        def repl_email(m):
            a = m.group(0)
            al = a.lower()
            counts["email"] += 1
            hard.add(al)
            if al in addr_owner:
                return addr_owner[al].email
            if al in ext_email:
                return ext_email[al]
            local, dom = al.split("@", 1)
            if dom == "enron.com" or dom.endswith(".enron.com"):
                p = self._register_person(name_from_address(al), aliases, hard, soft)
                return p.email if p else f"user{_h(al) % 1000}@{self.f.company.domain}"
            if dom in FREEMAIL:
                fake_local = f"{local[:1]}{_h(rng_key, al, salt=self.f.salt) % 9000 + 100}"
                return f"{fake_local}@{dom}"
            hard.add(dom)
            return f"{re.sub(r'[^a-z0-9]', '', local)[:1] or 'x'}{_h(rng_key, local, salt=self.f.salt) % 900 + 100}@{_fake_domain(self.f, dom)}"

        def repl_url(m):
            counts["url"] += 1
            u = m.group(0)
            dom = re.sub(r"^(?:https?://)?(?:www\.)?", "", u).split("/")[0].lower()
            hard.add(dom)
            return f"https://www.{_fake_domain(self.f, dom)}/"

        def repl_phone(m):
            counts["phone"] += 1
            digits = re.sub(r"\D", "", m.group(0))
            hard.add(digits[-10:])
            return self.f.fresh_phone_like(digits)

        def repl_ssn(m):
            counts["id"] += 1
            r = random.Random(_h(rng_key, m.group(0), salt=self.f.salt))
            return f"{r.randint(100, 899)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"

        def repl_num(m):
            counts["id"] += 1
            s = m.group(0)
            r = random.Random(_h(rng_key, s, salt=self.f.salt))
            return "".join(str(r.randint(1 if i == 0 else 0, 9)) for i in range(len(s)))

        def repl_zip(m):
            counts["loc"] += 1
            r = random.Random(_h(rng_key, m.group(0), salt=self.f.salt))
            return str(r.randint(10000, 99899))

        def repl_office(m):
            counts["loc"] += 1
            r = random.Random(_h(rng_key, m.group(0), salt=self.f.salt))
            return f"Room {r.randint(2, 30)}{r.choice('ABCD')}{r.randint(10, 99)}"

        ref_year = int(e.date[:4]) if e.date else 2001

        def _shift(y, mo, d):
            from datetime import date as _d, timedelta as _td
            try:
                return _d(y, mo, d) + _td(days=shift_days)
            except ValueError:
                return None

        def repl_ndate(m0):
            m = re.fullmatch(NDATE_RE, m0.group(0))
            mo, sep, d, y = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
            yy = int(y) if len(y) == 4 else (2000 + int(y) if int(y) < 50 else 1900 + int(y))
            n = _shift(yy, mo, d)
            if n is None or not shift_days or not (1990 <= yy <= 2003):
                return m.group(0)
            counts["date"] = counts.get("date", 0) + 1
            return f"{n.month}{sep}{n.day}{sep}{n.year}" if len(y) == 4 else f"{n.month}{sep}{n.day}{sep}{n.year % 100:02d}"

        def _ord(d, suf):
            if not suf:
                return str(d)
            return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"

        def repl_tdate(m0):
            m = re.fullmatch(TDATE_RE, m0.group(0))
            if not shift_days or m is None:
                return m0.group(0)
            mo, d, suf, y = _MON[m.group(1)[:3].lower()], int(m.group(2)), m.group(3), m.group(4)
            yy = int(y) if y else ref_year
            n = _shift(yy, mo, d)
            if n is None:
                return m.group(0)
            counts["date"] = counts.get("date", 0) + 1
            mon = m.group(1)
            if len(mon) > 3 or mon.endswith("."):
                mon = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][n.month - 1]
                mon = mon if len(m.group(1).rstrip(".")) > 3 else mon[:3]
            else:
                mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][n.month - 1]
            return f"{mon} {_ord(n.day, suf)}" + (f", {n.year}" if y else "")

        def repl_tdate2(m0):
            m = re.fullmatch(TDATE2_RE, m0.group(0))
            if not shift_days or m is None:
                return m0.group(0)
            d, suf, mon0, y = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            mo = _MON[mon0[:3].lower()]
            yy = int(y) if y else ref_year
            n = _shift(yy, mo, d)
            if n is None:
                return m.group(0)
            counts["date"] = counts.get("date", 0) + 1
            names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            mon = names[n.month - 1] if len(mon0) > 3 else names[n.month - 1][:3]
            return f"{_ord(n.day, suf)} {mon}" + (f" {n.year}" if y else "")

        alts: list[str] = [
            f"(?P<email>{EMAIL_RE})", f"(?P<zip>{ZIP_RE})", f"(?P<ndate>{NDATE_RE})", f"(?P<tdate>{TDATE_RE})", f"(?P<tdate2>{TDATE2_RE})", f"(?P<url>{URL_RE})", f"(?P<ssn>{SSN_RE})", f"(?P<phone>{PHONE_RE})",
            f"(?P<num>{LONGNUM_RE})", f"(?P<office>{OFFICE_RE})",
        ]
        ci_terms = sorted(ci_map, key=len, reverse=True)
        if ci_terms:
            alts.append(r"(?P<ci>(?i:(?<!\w)(?:" + "|".join(re.escape(t) for t in ci_terms) + r")(?!\w)))")
        if acr_map:
            alts.append(r"(?P<acr>(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(acr_map, key=len, reverse=True)) + r")(?!\w))")
        all_names = sorted(aliases, key=len, reverse=True)
        if all_names:
            alts.append(r"(?P<name>(?<!\w)(?:" + "|".join(re.escape(t).replace("\\ ", r"\s{1,3}") for t in all_names) + r")(?!\w)(?:" + NOTES_SUFFIX + r")?)")
        if ent_map:
            alts.append(r"(?P<ent>(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(ent_map, key=len, reverse=True)) + r")(?!\w))")
        master = re.compile("|".join(alts))

        def dispatch(m: re.Match) -> str:
            kind = m.lastgroup
            s = m.group(0)
            if kind == "email":
                return repl_email(m)
            if kind == "url":
                return repl_url(m)
            if kind == "zip":
                return repl_zip(m)
            if kind == "ndate":
                return repl_ndate(m)
            if kind == "tdate":
                return repl_tdate(m)
            if kind == "tdate2":
                return repl_tdate2(m)
            if kind == "ssn":
                return repl_ssn(m)
            if kind == "phone":
                return repl_phone(m)
            if kind == "num":
                return repl_num(m)
            if kind == "office":
                return repl_office(m)
            if kind == "ci":
                tag = ci_map[s.lower()]
                counts["company" if tag == "[COMPANY]" else ("org" if "ORG" in tag else "loc")] += 1
                return tag
            if kind == "acr":
                counts["unit"] += 1
                return acr_map[s]
            if kind == "name":
                base = re.sub(r"\s+", " ", re.split(r"/", s, maxsplit=1)[0])
                return aliases.get(base, base)
            if kind == "ent":
                return ent_map[s]
            return s

        sub = lambda t: master.sub(dispatch, t) if t else t
        # process trailer first so trailer addresses register into `addr_owner`-independent maps consistently
        new_subject, new_body, new_trailer = sub(subject), sub(body), sub(trailer)
        new_subject = re.sub(r"^\s*(RE|FW|FWD)\s*:\s*", lambda m: m.group(1).capitalize() + ": ", new_subject, flags=re.I)

        # collision guard: a hard term that is also part of a synthetic name / company is not a leak
        allowed = set(re.findall(r"[a-z]+", " ".join(p.full.lower() for p in self.f._people.values())))
        allowed |= set(re.findall(r"[a-z]+", (self.f.company.name + " " + self.f.company.short + " " + self.f.company.domain).lower()))
        hard = {t for t in hard if t and t not in allowed and len(t) >= 3}
        soft = {t for t in soft if t not in allowed and len(t) >= 3}

        return Scrubbed(
            company=self.f.company, subject=new_subject, body=new_body, trailer=new_trailer,
            from_person=from_p or self.f.person_for("Unknown Sender"), to_people=to_ps, cc_people=cc_ps,
            placeholders=placeholders, hard_terms=hard, soft_terms=soft, counts=counts,
            original_words=set(re.findall(r"[a-z']+", original_text.lower())),
            synthetic_names={v for v in aliases.values() if " " in v and "," not in v},
            keep_names=sorted({v for v in aliases.values() if " " in v and "," not in v
                               and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", new_subject + "\n" + new_body + "\n" + new_trailer)}),
        )


def _too_common_to_block(term: str, text: str) -> bool:
    """NER often tags ordinary words ('Time', 'Outsourcing', 'Memorandum') as ORG. Such terms still get a placeholder
    (the LLM re-invents them) but must not count as a leak when the LLM legitimately uses the word again."""
    return zipf_frequency(term.lower(), "en") >= 3.3 or _is_common_word(term, text)


def _is_common_word(term: str, text: str) -> bool:
    """True if `term` also occurs in lower case in the text (e.g. NER tagged 'Board' but 'board' is used normally)."""
    t = term.lower()
    return t == term.lower() and bool(re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", text)) if t != term else False


HEADER_LINE_RE = re.compile(r"^[ \t>]*(?:From|To|Cc|cc|CC|Bcc|Sent by)\s*:\s*(.+)$", re.M)
_LASTFIRST_RE = re.compile(r"^([A-Z][A-Za-z'\-]+),\s+([A-Z][A-Za-z'\-.]+(?: [A-Z]\.?)?)$")
_FIRSTLAST_RE = re.compile(r"^([A-Z][A-Za-z'\-.]+(?: [A-Z]\.?)? [A-Z][A-Za-z'\-]+)$")


def harvest_header_names(text: str) -> list[str]:
    """Names in From:/To:/cc: lines of quoted headers, in Outlook ('Last, First; Last, First') or
    Lotus Notes ('First Last/HOU/ECT@ECT, First Last/Corp/Enron') style."""
    out = []
    for m in HEADER_LINE_RE.finditer(text):
        line = re.sub(r"\[mailto:[^\]]*\]", "", m.group(1))
        line = re.sub(r"<[^>]*>", "", line).strip()
        if "@" in line and "/" not in line:
            continue
        frags = re.split(r";|(?<=[A-Za-z@])\s*,\s+(?=[A-Z][\w'\-. ]*(?:/|,|$|\s*<))", line) if "/" in line else re.split(r";", line)
        if len(frags) == 1 and "/" not in line and _LASTFIRST_RE.match(line.strip(' "\'')):
            frags = [line]
        for fr in frags:
            fr = re.sub(NOTES_SUFFIX, "", fr).strip(' \t"\'')
            if _LASTFIRST_RE.match(fr):
                out.append(fr)
            elif _FIRSTLAST_RE.match(fr) and not any(w.lower() in NER_STOPLIST for w in fr.split()):
                out.append(fr)
    return out


_BIGRAM_RE = re.compile(r"(?<![\w])([A-Z][a-z]{2,})(?: [A-Z]\.)? (?=([A-Z][a-z]{2,}(?:-[A-Z][a-z]+)?)(?![\w]))")


def harvest_capitalized_names(text: str) -> list[str]:
    """Safety net behind NER: any 'Firstname Lastname' where Firstname is a known given name (gender-guesser lexicon).
    Catches names that spaCy misses in all-caps / list / header contexts (e.g. inside Lotus Notes To: lists)."""
    out, seen = [], set()
    for m in _BIGRAM_RE.finditer(text):
        first, last = m.group(1), m.group(2)
        if first.lower() in NER_STOPLIST or last.lower() in NER_STOPLIST or first.lower() in _MONTHS_DAYS or last.lower() in _MONTHS_DAYS:
            continue
        if guess_gender(first) == "u" or last.lower() in COMMON_NON_SURNAMES:
            continue
        key = f"{first} {last}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _ws_join(text: str) -> str:
    """Like _ws but also joins single hard line-breaks, so a name wrapped over two lines ('Linda \\nGuinn') is one bigram."""
    return re.sub(r"(?<!\n)[ \t]*\n[ \t]*(?!\n)", " ", _ws(text))


_LOWER_BIGRAM_RE = re.compile(r"(?<![\w'])([a-z]{3,}) (?=([a-z]{3,})(?![\w']))")   # lookahead => overlapping bigrams


def harvest_lowercase_names(text: str) -> list[str]:
    """All-lowercase writers ('my paralegal, linda guinn, is ...'): known given name + a *rare* word (zipf < 3.3)."""
    out, seen = [], set()
    for m in _LOWER_BIGRAM_RE.finditer(text):
        first, last = m.group(1), m.group(2)
        if first in NER_STOPLIST or last in NER_STOPLIST or first in _MONTHS_DAYS or last in _MONTHS_DAYS or last in COMMON_NON_SURNAMES:
            continue
        if len(first) < 4 or guess_gender(first) == "u" or zipf_frequency(first, "en") > 5.2 or zipf_frequency(last, "en") >= 3.3:
            continue
        key = f"{first.capitalize()} {last.capitalize()}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


_MONTHS_DAYS = {"january", "february", "march", "april", "june", "july", "august", "september", "october", "november",
                "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
COMMON_NON_SURNAMES = {"street", "avenue", "road", "center", "energy", "power", "gas", "corp", "company", "group", "services",
                       "north", "south", "east", "west", "park", "house", "bank", "river", "lake", "hill", "valley", "beach"}

_ARTIFACT_SUBS = [
    (re.compile(r"\s*\[mailto:[^\]]*\]"), ""),                # Exchange 'mailto' blobs with encoded Enron addresses
    (re.compile(r"IMCEA[A-Z]+-[^\s\]>]+"), ""),
    (re.compile(r"(>)@(?:ENRON|Enron|enron)\b(?!\.)"), r"\1"),  # '<a@b.com>@ENRON'
    (re.compile(r"(@[\w-]+(?:\.[\w-]+)+)@(?:ENRON|Enron|enron)\b(?!\.)"), r"\1"),  # 'ernie@dynegy.com@ENRON'
]


_QP_CODES = {"20": " ", "3D": "=", "3d": "=", "92": "'", "93": '"', "94": '"', "96": "-", "09": " "}


def repair_quoted_printable(text: str) -> str:
    """Some Enron bodies contain quoted-printable debris ('janel= =20 guerrero', 'foo=\\n bar') that splits names."""
    if len(re.findall(r"=[0-9A-F]{2}", text)) < 2 and "= =" not in text:
        return text
    text = re.sub(r"=\s*=([0-9A-Fa-f]{2})\s*", lambda m: _QP_CODES.get(m.group(1), " ") if m.group(1) in ("20", "09") else _QP_CODES.get(m.group(1), " "), text)
    text = re.sub(r"=\n", "", text)
    return re.sub(r"=([0-9A-Fa-f]{2})", lambda m: _QP_CODES.get(m.group(1), m.group(0)), text)


def clean_artifacts(text: str) -> str:
    text = repair_quoted_printable(text)
    for pat, rep in _ARTIFACT_SUBS:
        text = pat.sub(rep, text)
    return text


def _ws(text: str) -> str:
    """Whitespace-normalised copy used only for *finding* names (never for output)."""
    text = re.sub(r"[ \t]+\n[ \t]*(?=[A-Z][\w'\-. ]*/)", " ", text)     # wrapped Lotus-Notes recipient lists
    return re.sub(r"[ \t]{2,}", " ", text)


def _trim_words(text: str, max_words: int) -> str:
    if not text or len(text.split()) <= max_words:
        return text
    out, n = [], 0
    for line in text.split("\n"):
        w = len(line.split())
        if n + w > max_words:
            break
        out.append(line)
        n += w
    return "\n".join(out)


def _fake_domain(factory: PersonaFactory, original_domain: str) -> str:
    r = random.Random(_h(factory.company.id, "dom", original_domain, salt=factory.salt))
    syll = ["nor", "vel", "tan", "mar", "dyn", "ora", "kel", "bri", "sol", "tri", "lum", "pex", "cor", "alt", "ven", "zen"]
    tld = original_domain.rsplit(".", 1)[-1] if original_domain.rsplit(".", 1)[-1] in {"com", "org", "net", "edu", "gov"} else "com"
    return f"{r.choice(syll)}{r.choice(syll)}{r.choice(['corp', 'group', 'tech', 'labs', 'co', ''])}.{tld}"


# ----------------------------------------------------------------------------- leak detection
@dataclass
class Leak:
    term: str
    severity: str  # 'hard' | 'soft'


def find_leaks(text: str, s: Scrubbed) -> list[Leak]:
    """Independent check: does any original identifier still appear in `text`?"""
    low = text.lower()
    out = []
    for t in sorted(s.hard_terms):
        if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low):
            out.append(Leak(t, "hard"))
    for t in sorted(s.soft_terms):
        if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low):
            out.append(Leak(t, "soft"))
    return out
