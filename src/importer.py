from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .models import Conversation, Message
from .utils import redact_phone


LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.*)$")

ROLE_MAP = {
    "agent": "agent",
    "customer": "customer",
    "caller": "customer",
    "bot": "bot",
    "system": "other",
}

DOMAIN_NOUNS = [
    "ivr",
    "call flow",
    "department",
    "agent",
    "user",
    "plan",
    "pricing",
    "invoice",
    "tds",
    "renewal",
    "billing",
    "whatsapp",
    "crm",
    "telecrm",
    "bitrix",
    "integration",
    "api",
    "webhook",
    "template",
    "otp",
    "dashboard",
    "panel",
    "number",
    "toll free",
    "missed call",
    "recording",
]

FILLER_WORDS = [
    "haan",
    "han",
    "ji",
    "hanji",
    "ok",
    "okay",
    "acha",
    "hmm",
    "hello",
    "hi",
    "good morning",
    "good evening",
    "thanks",
    "thank you",
    "sorry",
    "repeat",
    "dobara",
    "kya",
    "kahan",
]

FILLER_PHRASES = [
    "dobara boliye",
    "aap dobara",
    "sorry?",
    "kya bola",
    "awaz",
    "are you there",
]

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|sorry|ok|okay|sure|yep|yeah|haan|han|ji|hanji|acha|hmm)$",
    re.IGNORECASE,
)

NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")


def _normalize_text(text: str) -> str:
    cleaned = NON_WORD_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_domain_noun(text: str) -> bool:
    lowered = text.lower()
    return any(noun in lowered for noun in DOMAIN_NOUNS)


def _is_only_filler_words(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if normalized in FILLER_WORDS:
        return True
    tokens = normalized.split()
    return all(token in FILLER_WORDS for token in tokens)


def _is_filler(text: str) -> bool:
    if not text:
        return True
    if _contains_domain_noun(text):
        return False
    normalized = _normalize_text(text)
    if len(normalized) <= 30 and GREETING_PATTERNS.match(normalized):
        return True
    if _is_only_filler_words(text):
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in FILLER_PHRASES)


def read_raw_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data]
    return data


def parse_transcript(transcript: str) -> Tuple[List[Message], List[str]]:
    messages: List[Message] = []
    warnings: List[str] = []
    if not transcript:
        return messages, warnings

    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            if messages:
                messages[-1].text = f"{messages[-1].text} {line}".strip()
                warnings.append(f"Unparsed line appended: {line}")
            else:
                warnings.append(f"Unparsed line dropped: {line}")
            continue
        timestamp, speaker, text = match.groups()
        speaker_norm = speaker.strip().lower()
        role = ROLE_MAP.get(speaker_norm, "other")
        messages.append(Message(role=role, text=text.strip(), timestamp=timestamp))
    return messages, warnings


def normalize_record(raw: Dict[str, Any], redact_pii: bool = False) -> Conversation:
    call_id = raw.get("call_id") or raw.get("conversation_id") or "unknown"
    transcript = raw.get("call_transcript", "") or ""
    messages, warnings = parse_transcript(transcript)
    metadata = {k: v for k, v in raw.items() if k != "call_transcript"}

    if redact_pii:
        if "caller" in metadata and isinstance(metadata["caller"], str):
            metadata["caller"] = redact_phone(metadata["caller"])
        for msg in messages:
            msg.text = redact_phone(msg.text)

    for msg in messages:
        msg.is_filler = _is_filler(msg.text)

    messages_for_llm = [msg for msg in messages if not msg.is_filler]
    removed_filler_count = len(messages) - len(messages_for_llm)

    num_messages = len(messages_for_llm)
    total_text_chars = sum(len(m.text) for m in messages_for_llm)
    is_low_signal = num_messages < 4 or total_text_chars < 200

    metadata.update(
        {
            "is_low_signal": is_low_signal,
            "num_messages": num_messages,
            "total_text_chars": total_text_chars,
            "removed_filler_count": removed_filler_count,
            "parse_warnings": warnings,
        }
    )

    convo = Conversation(
        conversation_id=str(call_id),
        channel="call",
        messages=messages,
        messages_for_llm=messages_for_llm,
        metadata=metadata,
    )
    return convo


def normalize_records(rows: Iterable[Dict[str, Any]], redact_pii: bool = False) -> List[Dict[str, Any]]:
    normalized = []
    for raw in rows:
        convo = normalize_record(raw, redact_pii=redact_pii)
        normalized.append(convo.model_dump())
    return normalized
