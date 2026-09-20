"""Fictional target companies and deterministic synthetic people.

Design goals
------------
* Every company is fictional, Indian and in a different industry / city, so the synthetic corpus is
  varied (not "Enron with a different logo") but stays simple: Indian names, Rs./lakh/crore, Indian phone numbers, English text.
* Mapping original person -> synthetic person is a keyed hash, so:
    - the same Enron person always becomes the same synthetic person inside one company
      (thread / cross-email consistency, needed for tasks such as thread reconstruction), and
    - it is irreversible without the secret `salt`.
* Names/phones follow Indian formats (via Faker en_IN); the e-mail text stays in English. Set SYNTH_COMPANY_SET=international to use the earlier multi-country list (needed only to reproduce stored outputs).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import gender_guesser.detector as gender
from faker import Faker

_GD = gender.Detector(case_sensitive=False)


@dataclass(frozen=True)
class Company:
    id: str
    name: str            # legal / display name
    short: str           # how employees refer to it
    domain: str
    industry: str
    locale: str          # Faker locale for names / phones / addresses
    country: str
    city: str
    currency: str        # symbol used for money amounts
    currency_note: str   # e.g. "Indian rupees (₹, lakh/crore)"
    sub_units: tuple[str, ...] = ()   # plausible divisions, replace "ENA/ECT/EES"-style unit names
    hq_address: str = ""


INDIAN_COMPANIES: list[Company] = [   # default: fictional Indian companies, Indian names / phones / rupees, English text
    Company("agriindia", "Agriculture India Pvt. Ltd.", "AgriIndia", "agriindia.com", "agribusiness & commodity trading", "en_IN", "India", "Pune", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Procurement", "Commodities Desk", "Farm Partnerships")),
    Company("bharaturja", "Bharat Urja Grid Ltd", "Bharat Urja", "bharaturja.co.in", "power generation & transmission", "en_IN", "India", "Mumbai", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Grid Operations", "Power Trading", "Regulatory Affairs")),
    Company("kaveribank", "Kaveri Cooperative Bank Ltd", "Kaveri Bank", "kaveribank.co.in", "banking & lending", "en_IN", "India", "Chennai", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Credit Risk", "Treasury", "Corporate Lending")),
    Company("sahyadri", "Sahyadri Logistics Pvt. Ltd.", "Sahyadri", "sahyadrilogistics.in", "freight & supply-chain logistics", "en_IN", "India", "Nagpur", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Fleet Operations", "Customs", "Key Accounts")),
    Company("tiranga", "Tiranga Telecom Ltd", "Tiranga", "tirangatelecom.in", "telecommunications", "en_IN", "India", "New Delhi", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Network Engineering", "Enterprise Sales", "Regulatory")),
    Company("arogya", "Arogya Care Hospitals Ltd", "Arogya Care", "arogyacare.in", "hospital group & healthcare services", "en_IN", "India", "Hyderabad", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Clinical Operations", "Procurement", "Legal & Compliance")),
    Company("nilgiri", "Nilgiri Software Services Pvt. Ltd.", "Nilgiri Software", "nilgirisoft.in", "IT services & software", "en_IN", "India", "Bengaluru", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Engineering", "Delivery", "Sales Operations")),
    Company("annapurna", "Annapurna Foods Ltd", "Annapurna", "annapurnafoods.in", "food processing & FMCG", "en_IN", "India", "Ahmedabad", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Supply Chain", "Category Management", "Quality")),
    Company("himal", "Himal Steel & Alloys Ltd", "Himal Steel", "himalsteel.co.in", "steel & alloy manufacturing", "en_IN", "India", "Kolkata", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Plant Operations", "Procurement", "Contracts")),
    Company("gangautil", "Ganga Municipal Utilities", "Ganga Utilities", "gangautilities.gov.in", "municipal water & power utility", "en_IN", "India", "Lucknow", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Operations", "Customer Billing", "Legal")),
    Company("iyerkapoor", "Iyer & Kapoor Associates", "Iyer & Kapoor", "iyerkapoor.in", "law firm & corporate advisory", "en_IN", "India", "New Delhi", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Litigation", "Corporate", "Compliance")),
    Company("deccan", "Deccan Capital Partners", "Deccan Capital", "deccancapital.in", "investment & asset management", "en_IN", "India", "Mumbai", "Rs.", "Indian rupees (Rs.; lakh and crore are natural, e.g. Rs. 4.5 lakh, Rs. 12 crore)", ("Deal Team", "Portfolio Ops", "Investor Relations")),
]

import os as _os
COMPANIES: list[Company] = INTERNATIONAL_COMPANIES if _os.getenv("SYNTH_COMPANY_SET") == "international" else INDIAN_COMPANIES

INTERNATIONAL_COMPANIES: list[Company] = [   # earlier multi-country set; kept only to reproduce the stored results
    Company("agriindia", "Agriculture India Pvt. Ltd.", "AgriIndia", "agriindia.com", "agribusiness & commodity trading",
            "en_IN", "India", "Pune", "₹", "Indian rupees (₹; lakh / crore are natural)", ("Procurement", "Commodities Desk", "Farm Partnerships")),
    Company("nordwind", "Nordwind Energie AG", "Nordwind", "nordwind-energie.de", "renewable power & grid operator",
            "de_DE", "Germany", "Hamburg", "€", "euros (€)", ("Grid Operations", "Wholesale Trading", "Regulatory Affairs")),
    Company("halcyon", "Halcyon Mutual Bank", "Halcyon", "halcyonbank.co.uk", "retail & commercial banking",
            "en_GB", "United Kingdom", "Leeds", "£", "pounds sterling (£)", ("Credit Risk", "Treasury", "Corporate Lending")),
    Company("tasman", "Tasman Ridge Resources Ltd", "Tasman Ridge", "tasmanridge.com.au", "mining & bulk logistics",
            "en_AU", "Australia", "Perth", "A$", "Australian dollars (A$)", ("Mine Planning", "Rail & Port", "Commercial")),
    Company("brasilog", "BrasiLog Transportes S.A.", "BrasiLog", "brasilog.com.br", "freight & supply-chain logistics",
            "pt_BR", "Brazil", "São Paulo", "R$", "Brazilian reais (R$)", ("Fleet Operations", "Customs", "Key Accounts")),
    Company("maple", "Maple & Crown Telecom Inc.", "Maple & Crown", "maplecrown.ca", "telecommunications",
            "en_CA", "Canada", "Calgary", "C$", "Canadian dollars (C$)", ("Network Engineering", "Enterprise Sales", "Regulatory")),
    Company("lumiere", "Lumière Santé SAS", "Lumière Santé", "lumieresante.fr", "private hospital group & healthcare services",
            "fr_FR", "France", "Lyon", "€", "euros (€)", ("Clinical Operations", "Procurement", "Legal & Compliance")),
    Company("cedarpoint", "Cedar Point Capital Partners", "Cedar Point", "cedarpointcap.com", "private equity & asset management",
            "en_US", "United States", "Charlotte", "$", "US dollars ($)", ("Deal Team", "Portfolio Ops", "Investor Relations")),
    Company("orion", "Orion Ridge Software Ltd", "Orion Ridge", "orionridge.ie", "enterprise software & SaaS",
            "en_IE", "Ireland", "Dublin", "€", "euros (€)", ("Engineering", "Customer Success", "Sales Ops")),
    Company("solmar", "Solmar Foods S.L.", "Solmar", "solmarfoods.es", "food processing & retail distribution",
            "es_ES", "Spain", "Valencia", "€", "euros (€)", ("Supply Chain", "Category Management", "Quality")),
    Company("kestrel", "Kestrel Aerospace Components Ltd", "Kestrel", "kestrelaero.co.uk", "aerospace manufacturing",
            "en_GB", "United Kingdom", "Bristol", "£", "pounds sterling (£)", ("Manufacturing", "Supplier Quality", "Contracts")),
    Company("lakeshore", "Lakeshore Municipal Utilities", "Lakeshore Utilities", "lakeshoreutilities.gov", "municipal water & power utility",
            "en_US", "United States", "Cleveland", "$", "US dollars ($)", ("Operations", "Customer Billing", "Legal")),
    Company("meridian", "Meridian Law & Advisory LLP", "Meridian", "meridianlaw.com", "law firm & corporate advisory",
            "en_GB", "United Kingdom", "London", "£", "pounds sterling (£)", ("Litigation", "Corporate", "Compliance")),
]
COMPANY_BY_ID = {c.id: c for c in INDIAN_COMPANIES + INTERNATIONAL_COMPANIES}


PHONE_FMTS = {
    "en_IN": ["+91 98### #####", "+91-20-####-####", "+91-11-####-####"],
    "de_DE": ["+49 40 ### ####", "+49 171 ### ####", "+49 30 #### ###"],
    "en_GB": ["+44 113 ### ####", "+44 20 #### ####", "+44 7### ######"],
    "en_AU": ["+61 8 #### ####", "+61 4## ### ###"],
    "pt_BR": ["+55 11 9#### ####", "+55 11 #### ####"],
    "en_CA": ["(403) ###-####", "+1 403 ###-####", "403-###-####"],
    "fr_FR": ["+33 4 ## ## ## ##", "+33 6 ## ## ## ##"],
    "en_US": ["(704) ###-####", "704-###-####", "(216) ###-####", "216.###.####"],
    "en_IE": ["+353 1 ### ####", "+353 87 ### ####"],
    "es_ES": ["+34 96 ### ## ##", "+34 6## ### ###"],
}


def _h(*parts: str, salt: str = "") -> int:
    return int(hashlib.sha256(("|".join(parts) + "|" + salt).encode()).hexdigest(), 16)


def pick_company(key: str, salt: str = "", companies: list[Company] | None = None) -> Company:
    """Deterministically map a key (e.g. the Enron mailbox owner) to a company."""
    cs = companies or COMPANIES
    return cs[_h(key, salt=salt) % len(cs)]


def guess_gender(first_name: str) -> str:
    g = _GD.get_gender(first_name.split()[0].capitalize()) if first_name else "unknown"
    if g in ("male", "mostly_male"):
        return "m"
    if g in ("female", "mostly_female"):
        return "f"
    return "u"


@dataclass
class Person:
    first: str
    last: str
    email: str
    gender: str
    title: str = ""
    phone: str = ""

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"


_TITLES = ["Manager", "Senior Analyst", "Director", "Vice President", "Associate", "Coordinator", "Head of Operations",
           "Counsel", "Analyst", "Specialist", "Executive Assistant", "Principal"]


class PersonaFactory:
    """Creates and caches synthetic people for one target company."""

    def __init__(self, company: Company, salt: str = "synth-v1"):
        self.company = company
        self.salt = salt
        self._people: dict[str, Person] = {}
        self._fakers: dict[int, Faker] = {}
        self._used_emails: set[str] = set()

    _FAKERS: dict[str, Faker] = {}   # Faker() construction is slow; one per locale, re-seeded per call

    def _faker(self, seed: int) -> Faker:
        f = PersonaFactory._FAKERS.get(self.company.locale)
        if f is None:
            f = PersonaFactory._FAKERS[self.company.locale] = Faker(self.company.locale)
        f.seed_instance(seed % (2 ** 32))
        return f

    @staticmethod
    def _ascii(s: str) -> str:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).encode("ascii", "ignore").decode()

    def person_for(self, original_name: str, gender_hint: str | None = None) -> Person:
        """Deterministic synthetic person for an original (real) name."""
        key = " ".join(original_name.lower().split())
        if key in self._people:
            return self._people[key]
        seed = _h(self.company.id, key, salt=self.salt)
        f = self._faker(seed)
        g = gender_hint or guess_gender(original_name.split()[0] if original_name else "")
        if g == "m":
            first = f.first_name_male()
        elif g == "f":
            first = f.first_name_female()
        else:
            first = f.first_name()
        last = f.last_name()
        base = f"{self._ascii(first).lower()}.{self._ascii(last).lower()}".replace(" ", "").replace("'", "")
        email, n = f"{base}@{self.company.domain}", 2
        while email in self._used_emails:
            email = f"{base}{n}@{self.company.domain}"
            n += 1
        self._used_emails.add(email)
        p = Person(first, last, email, g, title=random.Random(seed).choice(_TITLES), phone=self.phone(seed))
        self._people[key] = p
        return p

    def phone(self, seed: int | None = None) -> str:
        f = self._faker(seed if seed is not None else random.getrandbits(32))
        fmts = PHONE_FMTS.get(self.company.locale)
        if not fmts:
            return f.phone_number()
        return f.numerify(random.Random(seed).choice(fmts))

    def fresh_phone_like(self, original: str) -> str:
        """Phone in the target locale, seeded by the original so the same number maps to the same fake."""
        return self.phone(_h(self.company.id, original, salt=self.salt))

    def org_name(self, original: str) -> str:
        f = self._faker(_h(self.company.id, "org", original.lower(), salt=self.salt))
        return f.company()

    def city_name(self, original: str) -> str:
        f = self._faker(_h(self.company.id, "city", original.lower(), salt=self.salt))
        return f.city()

    def street_address(self, original: str) -> str:
        f = self._faker(_h(self.company.id, "addr", original.lower(), salt=self.salt))
        return f.street_address()
