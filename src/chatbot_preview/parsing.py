from __future__ import annotations

import re
from typing import Dict, Optional


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\+?\d[\d\s\-]{6,}\d\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s?(?:am|pm)?\b", re.IGNORECASE)


def extract_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    if not text:
        return fields
    email = EMAIL_RE.search(text)
    if email:
        fields["email"] = email.group(0)
    phone = PHONE_RE.search(text)
    if phone:
        fields["phone"] = phone.group(0)
    date = DATE_RE.search(text)
    if date:
        fields["date"] = date.group(0)
    time = TIME_RE.search(text)
    if time:
        fields["time"] = time.group(0)
    return fields


def guess_field_value(text: str, field_name: str) -> Optional[str]:
    if not text:
        return None
    name = (field_name or "").lower()
    extracted = extract_fields(text)
    if "email" in name or "mail" in name:
        return extracted.get("email") or text.strip()
    if "phone" in name or "mobile" in name or "contact" in name:
        return extracted.get("phone") or text.strip()
    if "date" in name or "time" in name or "schedule" in name:
        return extracted.get("date") or extracted.get("time") or text.strip()
    return text.strip()
