"""Parse raw Enron messages (RFC-822 text in the Kaggle `emails.csv`) into a structured form
and back into a realistic raw email.

An email is split into:
  * headers  (From / To / Cc / Date / Subject + a few X-* fields)
  * body     (the *new* text written by the sender)
  * trailer  (forwarded / quoted-reply block that follows the new text)

Keeping the trailer separate matters: ~1/3 of Enron mail contains forwarded or quoted
history, and a downstream LLM must see that structure in realistic form.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from email import policy
from email.parser import Parser
from email.utils import getaddresses, parsedate_to_datetime
from typing import Optional

# Markers that start a forwarded / quoted block. Order matters little; we take the earliest.
_TRAILER_PATTERNS = [
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.I | re.M),
    re.compile(r"^\s*-{5,}\s*Forwarded by .*$", re.I | re.M),
    re.compile(r"^\s*-{2,}\s*Forwarded (?:message|by).*$", re.I | re.M),
    re.compile(r"^\s*_{5,}\s*$", re.M),
    re.compile(r"^\s*On .{5,80} wrote:\s*$", re.M),
    re.compile(r"^\s*>+ ", re.M),
]

_EMAIL_RE = re.compile(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+")


@dataclass
class ParsedEmail:
    message_id: str = ""
    date: Optional[str] = None            # ISO string if parseable
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    cc_addrs: list[str] = field(default_factory=list)
    bcc_addrs: list[str] = field(default_factory=list)
    subject: str = ""
    x_from: str = ""
    x_to: str = ""
    x_folder: str = ""
    x_origin: str = ""
    body: str = ""       # new content
    trailer: str = ""    # forwarded / quoted history (verbatim)
    raw_body: str = ""   # body + trailer
    file: str = ""       # e.g. allen-p/_sent_mail/1.

    @property
    def folder_type(self) -> str:
        f = (self.file or "").lower()
        parts = f.split("/")
        return parts[1] if len(parts) > 1 else "unknown"


def _addr_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [a.lower() for _, a in getaddresses([value.replace("\n", " ").replace("\t", " ")]) if a]


def split_trailer(text: str) -> tuple[str, str]:
    """Split raw body into (new_text, trailer)."""
    cut = None
    for pat in _TRAILER_PATTERNS:
        m = pat.search(text)
        if m and (cut is None or m.start() < cut):
            cut = m.start()
    if cut is None:
        return text.rstrip(), ""
    return text[:cut].rstrip(), text[cut:].strip("\n")


def parse_email(raw: str, file: str = "") -> ParsedEmail:
    msg = Parser(policy=policy.compat32).parsestr(raw)
    raw_body = msg.get_payload() if not msg.is_multipart() else ""
    if not isinstance(raw_body, str):
        raw_body = str(raw_body)
    date_iso = None
    try:
        date_iso = parsedate_to_datetime(msg["Date"]).isoformat() if msg["Date"] else None
    except (TypeError, ValueError):
        pass
    body, trailer = split_trailer(raw_body.replace("\r\n", "\n"))
    return ParsedEmail(
        message_id=(msg["Message-ID"] or "").strip(),
        date=date_iso,
        from_addr=(_addr_list(msg["From"]) or [""])[0],
        to_addrs=_addr_list(msg["To"]),
        cc_addrs=_addr_list(msg["Cc"]),
        bcc_addrs=_addr_list(msg["Bcc"]),
        subject=(msg["Subject"] or "").replace("\n", " ").strip(),
        x_from=(msg["X-From"] or "").strip(),
        x_to=(msg["X-To"] or "").strip(),
        x_folder=(msg["X-Folder"] or "").strip(),
        x_origin=(msg["X-Origin"] or "").strip(),
        body=body,
        trailer=trailer,
        raw_body=raw_body,
        file=file,
    )


def find_emails(text: str) -> list[str]:
    return _EMAIL_RE.findall(text or "")


def render_email(p: ParsedEmail, with_headers: bool = True) -> str:
    """Render a ParsedEmail back into a plain RFC-822-like text."""
    lines = []
    if with_headers:
        lines += [
            f"Message-ID: {p.message_id}",
            f"Date: {p.date or ''}",
            f"From: {p.from_addr}",
            f"To: {', '.join(p.to_addrs)}",
        ]
        if p.cc_addrs:
            lines.append(f"Cc: {', '.join(p.cc_addrs)}")
        lines.append(f"Subject: {p.subject}")
        lines.append("")
    lines.append(p.body)
    if p.trailer:
        lines += ["", p.trailer]
    return "\n".join(lines)
