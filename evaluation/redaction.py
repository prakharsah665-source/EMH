"""
PII redaction applied AT CAPTURE TIME.

Raw page bodies and captured transcript text may contain
candidate PII (name banners, emails, phone numbers) and secrets
(the interview JWT in URLs). Everything that flows into
artifacts, stdout (and therefore the JUnit XML / HTML reports,
which embed captured stdout) must pass through redact_pii()
first.

Deliberately conservative: interview words are never dropped,
only masked spans are replaced, so WER/judging on redacted text
stays meaningful.
"""

import re


_JWT = re.compile(
    r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"
)
_EMAIL = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
# Phone-like: digits with typical phone separators (no dot, so
# decimals/versions like "1.0.25" survive), optional +country.
# The candidate span must contain >= 8 digits to be masked.
_PHONE = re.compile(
    # Trailing guard rejects a following word char or a dot
    # that starts another digit (decimal/version segments),
    # while allowing an ordinary sentence-ending period.
    r"(?<![\w.])\+?\d[\d\s()-]{5,}\d(?!\w)(?!\.\d)"
)
# Bare long digit runs (ids, card-like numbers).
_DIGIT_RUN = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def _mask_phone(match: re.Match) -> str:
    span = match.group(0)
    digits = sum(ch.isdigit() for ch in span)
    return "[REDACTED-PHONE]" if digits >= 8 else span


def redact_pii(text: str) -> str:
    """Mask JWTs, emails and phone-like/long digit runs."""

    if not text:
        return text

    text = _JWT.sub("[REDACTED-JWT]", text)
    text = _EMAIL.sub("[REDACTED-EMAIL]", text)
    text = _PHONE.sub(_mask_phone, text)
    text = _DIGIT_RUN.sub("[REDACTED-DIGITS]", text)
    return text
