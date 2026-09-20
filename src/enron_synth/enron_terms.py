"""Enron-specific world knowledge used to (a) scrub and (b) *verify* that nothing identifying survives.

These are "quasi-identifiers": even after every person name is replaced, a string like `EOL`,
`Raptor` or `/HOU/ECT` immediately tells a reader (or an LLM that memorised the corpus) where the
email came from.
"""

# The company itself -> [COMPANY]
COMPANY_TERMS = ["Enron", "EnronOnline", "Enron Online", "Enron Corp", "Enron Corporation", "Enron North America",
                 "Enron Energy Services", "Enron Capital & Trade", "Enron Broadband", "Enron Networks",
                 "Enron Wholesale", "Enron Global Markets", "Enron Industrial Markets", "Enron Building"]

# Internal business-unit / system acronyms (matched case-sensitively, whole word) -> [UNIT_n]
UNIT_ACRONYMS = ["ENA", "ECT", "EES", "EWS", "EGM", "EBS", "EIM", "EEOS", "EOL", "ECS", "EGS", "HPL", "NNG",
                 "EESI", "ERMS", "ETS", "EE&CC", "EPMI", "EI"]

# Well-known Enron-world organisations / vehicles / people-adjacent terms (case-insensitive) -> [ORG_n]
ORG_TERMS = [
    "Arthur Andersen", "Andersen", "Vinson & Elkins", "Vinson and Elkins", "Dynegy", "Azurix", "Portland General",
    "Northern Natural Gas", "Transwestern", "Transwestern Pipeline", "Northern Border", "Chewco", "LJM", "LJM2",
    "Raptor", "Raptors", "JEDI", "JEDI II", "Whitewing", "Braveheart", "Sithe",
    "Calpine", "Mirant", "Williams Energy", "El Paso", "Duke Energy", "Sempra", "Enserch", "Kinder Morgan",
    "PG&E", "Pacific Gas & Electric", "Southern California Edison", "SoCal Edison", "SoCalGas", "Cal PX", "CalPX",
    "CAISO", "Cal ISO", "California ISO", "Bonneville Power", "BPA", "Portland General Electric", "PGE",
    "Nevada Power", "Sierra Pacific", "Puget Sound", "Avista", "Powerex", "Merrill Lynch", "Citigroup", "Citibank",
    "JP Morgan", "JPMorgan", "Credit Suisse", "CSFB", "Bank of America", "Goldman Sachs",
    "Lehman Brothers", "Morgan Stanley", "UBS", "Barclays", "Deutsche Bank", "Bear Stearns", 
]

# Locations strongly tied to Enron -> [LOC_n]
LOC_TERMS = ["Houston", "Enron Center", "Smith Street", "Three Allen Center", "Allen Center", "Harris County",
             "Sugar Land", "The Woodlands", "Kingwood"]

# Surnames that are ordinary English words: never replace them when they appear alone (too many false positives)
COMMON_WORD_SURNAMES = {
    "price", "hall", "ward", "white", "black", "brown", "green", "young", "king", "long", "short", "love", "may",
    "mark", "will", "bill", "grant", "rose", "day", "hill", "lane", "wood", "fair", "cook", "baker", "page", "strong",
    "stone", "field", "west", "east", "north", "south", "case", "best", "little", "hope", "sweet", "cross", "bank",
    "gold", "bell", "block", "brooks", "carter", "cash", "clear", "cole", "dawson", "dean", "ford", "frank", "fox",
    "gay", "gray", "hunt", "hart", "jones", "lamb", "lee", "major", "miller", "mills", "moore", "nelson", "parks",
    "pierce", "rice", "rich", "ross", "sanders", "scott", "smart", "swift", "taylor", "tucker", "wall", "warren",
    "watson", "webb", "wells", "wilde", "wright",
}

# Tokens NER often mislabels as PERSON/ORG/LOC in email text
NER_STOPLIST = {
    "fyi", "pst", "est", "cst", "cdt", "edt", "pdt", "asap", "pdf", "xls", "doc", "ok", "re", "fw", "fwd", "cc", "bcc",
    "thanks", "thank", "regards", "best", "hi", "hello", "dear", "please", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december", "am", "pm", "mr", "ms", "mrs", "dr", "inc", "corp", "llc", "ltd", "co", "us",
    "usa", "u.s.", "the", "to", "from", "subject", "sent", "original message", "forwarded", "attached", "yes", "no",
    "gas", "power", "email", "e-mail", "internet", "web", "excel", "word", "outlook", "lotus notes",
}

# Nickname -> formal (used so that "Bill" is treated as the same person as "William")
NICKNAMES = {
    "bill": "william", "will": "william", "bob": "robert", "rob": "robert", "mike": "michael", "jim": "james",
    "tom": "thomas", "dave": "david", "steve": "steven", "chris": "christopher", "rick": "richard", "rich": "richard",
    "jeff": "jeffrey", "phil": "phillip", "dan": "daniel", "matt": "matthew", "tim": "timothy", "joe": "joseph",
    "jon": "jonathan", "sue": "susan", "kate": "katherine", "liz": "elizabeth", "andy": "andrew", "tony": "anthony",
    "ed": "edward", "greg": "gregory", "ken": "kenneth", "don": "donald", "ron": "ronald", "larry": "lawrence",
    "pete": "peter", "sam": "samuel", "charlie": "charles", "chuck": "charles", "jenny": "jennifer", "jen": "jennifer",
    "kim": "kimberly", "ben": "benjamin", "nick": "nicholas", "pat": "patrick", "tommy": "thomas", "vince": "vincent",
    "debbie": "deborah", "deb": "deborah", "becky": "rebecca", "cathy": "catherine", "sandy": "sandra",
}

# Surnames of Enron-world figures that are unambiguous: if one appears in a *synthetic* e-mail it is a leak,
# whether or not it was in the source e-mail (LLMs have memorised the Enron story).
FAMOUS_SURNAMES = ["Skilling", "Fastow", "Causey", "Kopper", "Glisan", "Delainey", "Whalley", "Lavorato", "Kaminski",
                   "Haedicke", "Belden", "Shankman", "Sherriff", "Watkins", "Mintz", "McMahon"]
