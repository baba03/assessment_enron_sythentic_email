"""Offline tests (no API key needed): the deterministic parts of the pipeline."""
import re
from pathlib import Path

from enron_synth.parsing import parse_email, split_trailer
from enron_synth.personas import COMPANY_BY_ID, PersonaFactory
from enron_synth.pii import Scrubber, find_leaks
from enron_synth.pipeline import SyntheticEmailGenerator, rewrap, shift_date, unwrap, wrap_profile

RAW = (Path(__file__).parent / "sample_email.txt").read_text()
COMPANY = COMPANY_BY_ID["agriindia"]


def scrub(shift_days=0):
    e = parse_email(RAW, "fastow-a/sent/1.")
    return e, Scrubber(PersonaFactory(COMPANY)).scrub(e, shift_days=shift_days)


def test_parse_headers_and_body():
    e = parse_email(RAW, "fastow-a/sent/1.")
    assert e.from_addr == "william.giuliani@enron.com"
    assert e.to_addrs == ["andrew.fastow@enron.com"] and e.cc_addrs == ["mark.haedicke@enron.com"]
    assert e.body.startswith("Dear Andrew") and e.trailer == ""


def test_trailer_split():
    body, trailer = split_trailer("hi\n\n-----Original Message-----\nFrom: x\n")
    assert body == "hi" and trailer.startswith("-----Original Message")


def test_no_original_identifiers_survive_scrub():
    _, s = scrub()
    text = "\n".join([s.subject, s.body, s.trailer])
    for term in ["Giuliani", "Fastow", "Haedicke", "Cline", "Enron", "Houston", "9048", "Beyer", "Gresham"]:
        assert term.lower() not in text.lower(), term
    assert not [l for l in find_leaks(text, s) if l.severity == "hard"]


def test_nickname_maps_to_same_person():
    _, s = scrub()
    assert s.from_person.full in s.body            # signature "Bill Giuliani" == header "William Giuliani"


def test_placeholders_only_for_things_llm_must_invent():
    _, s = scrub()
    assert "[COMPANY]" in s.body and set(re.findall(r"\[[A-Z]+_?\d*\]", s.body)) >= {"[COMPANY]"}


def test_determinism_and_cross_email_consistency():
    _, a = scrub()
    _, b = scrub()
    assert a.body == b.body and a.from_person == b.from_person
    f = PersonaFactory(COMPANY)
    assert f.person_for("Andrew Fastow") is f.person_for("andrew  fastow")


def test_different_companies_give_different_people():
    e = parse_email(RAW, "x/y/1.")
    a = Scrubber(PersonaFactory(COMPANY_BY_ID["agriindia"])).scrub(e)
    b = Scrubber(PersonaFactory(COMPANY_BY_ID["bharaturja"])).scrub(e)
    assert a.from_person.full != b.from_person.full and a.from_person.email.endswith("agriindia.com")


def test_date_shift_preserves_weekday_and_clock():
    dt, days = shift_date("2001-06-07T07:48:00-07:00", "fastow-a", COMPANY, "s")
    assert days % 7 == 0 and dt.weekday() == 3 and (dt.hour, dt.minute) == (7, 48) and dt.year in (2024, 2025)


def test_dates_in_text_shift_by_same_offset():
    e = parse_email(RAW.replace("Thank you.", "Call me on 6/12/2001 or Jun 14."), "a/b/1.")
    s = Scrubber(PersonaFactory(COMPANY)).scrub(e, shift_days=8400)
    assert "6/11/2024" in s.body and "Jun 13" in s.body        # +8400 days = 1200 weeks (weekday preserved)


def test_wrap_roundtrip():
    txt = ("This is a long paragraph that was hard wrapped at about seventy five \n"
           "columns by an old mail client, and it continues here with more words \n"
           "to fill the second line of the paragraph completely ok, and a third \n"
           "line of text to make it look wrapped, plus one more line for luck.\n\nShort.")
    w = wrap_profile(txt)
    assert w and "\n" not in unwrap(txt, w).split("\n\n")[0]
    assert max(len(l) for l in rewrap(unwrap(txt, w), w).split("\n")) <= w


def test_rule_mode_end_to_end_offline():
    out = SyntheticEmailGenerator().generate(RAW, "fastow-a/sent/1.", company="agriindia", mode="rule")
    assert out.headers["From"].endswith("@agriindia.com>") and "enron" not in out.raw.lower()
    assert not out.meta["leaks_after"] and "[" not in out.body


def test_lotus_lists_with_irregular_whitespace_and_qp_debris():
    """Regression: names in recipient lists with double spaces, wrapped lines and '= =20' debris survived scrubbing."""
    raw = ("Message-ID: <1>\nDate: Mon, 4 Sep 2000 09:00:00 -0700 (PDT)\nFrom: jeff.dasovich@enron.com\nTo: susan.mara@enron.com\n"
           "Subject: FW: plan\n\nsee below\n\n---------------------- Forwarded by Jeff Dasovich/NA/Enron on 09/04/2000 09:00 AM ---------------------------\n\n"
           "To: Susan J Mara/NA/Enron@ENRON, Alan Comnes/PDX/ECT@ECT, Linda  Guinn/HOU/ECT@ECT, Janel= =20 Guerrero/Corp/Enron@Enron, Jeff \n"
           "Dasovich/NA/Enron@Enron\nSubject: plan\n\nText.\n")
    e = parse_email(raw, "dasovich-j/sent/1.")
    s = Scrubber(PersonaFactory(COMPANY)).scrub(e, shift_days=8400)
    text = (s.subject + s.body + s.trailer).lower()
    for n in ["comnes", "guinn", "guerrero", "dasovich", "mara"]:
        assert n not in text, n
    assert "/hou/" not in text and "@ect" not in text


def test_name_wrapped_over_two_lines_and_lowercase_names():
    raw = ("Message-ID: <2>\nDate: Mon, 4 Sep 2000 09:00:00 -0700 (PDT)\nFrom: a.b@enron.com\nTo: c.d@enron.com\nSubject: x\n\n"
           "My paralegal, Linda \nGuinn, is the best contact. also ask rita  wynne about it.\n")
    s = Scrubber(PersonaFactory(COMPANY)).scrub(parse_email(raw, "x/y/1."))
    assert "guinn" not in s.body.lower() and "wynne" not in s.body.lower()


def test_signature_check():
    from enron_synth.pipeline import signature_problem
    assert signature_problem("Thanks,\nAdvik", "Thanks,\nEugene", "Advik")            # signed by someone else -> problem
    assert signature_problem("Thanks,\nAdvik", "Best regards,\nAdvik Kara", "Advik") is None
    assert signature_problem("please call me", "ok call", "Advik") is None             # source not signed by sender -> no constraint


def test_zip_dashdates_and_double_at_artifacts():
    raw = ("Message-ID: <3>\nDate: Mon, 4 Sep 2000 09:00:00 -0700 (PDT)\nFrom: a.b@enron.com\nTo: c.d@enron.com\nSubject: x\n\n"
           "Houston, TX 77002. See Manifesto 1-25-01.doc\n\n -----Original Message-----\nFrom: \tErnie.Kohnke@dynegy.com@ENRON\nSent: today\n")
    s = Scrubber(PersonaFactory(COMPANY)).scrub(parse_email(raw, "x/y/1."), shift_days=8400)
    text = s.body + s.trailer
    assert "77002" not in text and "1-25-01" not in text and "@ENRON" not in text and "1-25-24" in text


def test_restore_style_undoes_llm_typography():
    from enron_synth.pipeline import restore_style
    ref = ("First sentence here.  Second sentence there.  Third one follows.  Fourth is short.\n"
           "a long wrapped line of text that goes on and on and on for a while, \nand continues here.")
    out = restore_style("It\u2019s done \u2013 really. Next point. Another one. Last.", ref)
    assert "\u2019" not in out and "\u2013" not in out and "really.  Next" in out and out.isascii()
    assert restore_style("plain text. Fine.", "no double spaces. at all. here.") == "plain text. Fine."
