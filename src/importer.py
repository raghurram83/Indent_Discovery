from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .models import Conversation, Message
from .utils import redact_phone


LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.*)$")

LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.*)$")

BATCH_REC_RE = re.compile(r"^Recording\s+#(?P<num>\d+):\s*(?P<id>.+)$")
BATCH_URL_RE = re.compile(r"^URL:\s*(?P<url>.+)$", re.IGNORECASE)
BATCH_TS_RE = re.compile(r"^Timestamp:\s*(?P<ts>.+)$", re.IGNORECASE)
BATCH_SEP_RE = re.compile(r"^=+$")

ROLE_MAP = {
    "agent": "agent",
    "customer": "customer",
    "caller": "customer",
    "bot": "bot",
    "system": "other",
}

STRUCTURED_ROLE_MAP = {
    "user": "customer",
    "customer": "customer",
    "caller": "customer",
    "agent": "agent",
    "agnt": "agent",
    "bot": "bot",
    "system": "other",
}

CALL_ID_ALIAS_NORMALIZED = {"callid", "conversationid", "conversation", "id"}


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def _resolve_call_id(raw: Dict[str, Any]) -> str | None:
    for canonical in ("call_id", "conversation_id", "Conversation_id", "conversationId", "ConversationId", "id"):
        if canonical in raw:
            return raw[canonical]
    for key, value in raw.items():
        if _normalize_key(key) in CALL_ID_ALIAS_NORMALIZED:
            return value
    return None


def _resolve_structured_role(role_key: str) -> str:
    normalized = _normalize_key(role_key)
    return STRUCTURED_ROLE_MAP.get(normalized, "other")

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


def _parse_batch_transcription(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    transcript_lines: List[str] = []

    def flush() -> None:
        nonlocal current, transcript_lines
        if current is None:
            return
        transcript = "\n".join([ln.strip() for ln in transcript_lines if ln.strip()])
        if transcript:
            row: Dict[str, Any] = {
                "call_id": current.get("call_id") or current.get("recording_id") or current.get("recording_num"),
                "call_transcript": transcript,
            }
            if current.get("recording_url"):
                row["recording_url"] = current["recording_url"]
            if current.get("recording_timestamp"):
                row["recording_timestamp"] = current["recording_timestamp"]
            rows.append(row)
        current = None
        transcript_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if BATCH_SEP_RE.match(line):
            continue

        rec_match = BATCH_REC_RE.match(line)
        if rec_match:
            flush()
            rec_id_raw = rec_match.group("id").strip()
            rec_id = rec_id_raw.rsplit(".", 1)[0] if rec_id_raw else ""
            current = {
                "recording_num": rec_match.group("num"),
                "recording_id": rec_id_raw,
                "call_id": rec_id or rec_id_raw,
            }
            continue

        url_match = BATCH_URL_RE.match(line)
        if url_match and current is not None:
            current["recording_url"] = url_match.group("url").strip()
            continue

        ts_match = BATCH_TS_RE.match(line)
        if ts_match and current is not None:
            current["recording_timestamp"] = ts_match.group("ts").strip()
            continue

        if current is not None:
            transcript_lines.append(line)

    flush()
    return rows


def read_raw_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    if suffix == ".txt":
        text = path.read_text().strip()
        if not text:
            return []
        batch_rows = _parse_batch_transcription(text)
        if batch_rows:
            return batch_rows
        return [{"call_id": path.stem, "call_transcript": text}]

    data = json.loads(path.read_text())
    if isinstance(data, dict):
        conversations = data.get("conversations")
        if isinstance(conversations, list):
            base_metadata = {k: v for k, v in data.items() if k != "conversations"}
            rows: List[Dict[str, Any]] = []
            for convo in conversations:
                if not isinstance(convo, dict):
                    continue
                merged = {**base_metadata, **convo}
                rows.append(merged)
            return rows
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


def _cleanup_entry_text(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _format_structured_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return _cleanup_entry_text(entry)
    if isinstance(entry, dict):
        text_info = entry.get("text")
        if isinstance(text_info, dict):
            content = text_info.get("content")
            if isinstance(content, str) and content.strip():
                return _cleanup_entry_text(content)
        button = entry.get("url_button") or entry.get("button")
        if isinstance(button, dict):
            label = button.get("label")
            url = button.get("url")
            if label and url:
                return _cleanup_entry_text(f"{label} ({url})")
            if label:
                return _cleanup_entry_text(label)
            if url:
                return _cleanup_entry_text(url)
    if entry is not None:
        return _cleanup_entry_text(json.dumps(entry, ensure_ascii=False))
    return ""


def _parse_structured_transcript(transcript: Dict[str, Any]) -> Tuple[List[Message], List[str]]:
    messages: List[Message] = []
    warnings: List[str] = []
    for role_key, payload in transcript.items():
        if not payload:
            continue
        adapted_role = _resolve_structured_role(role_key)
        raw_segments: List[Any] = []
        if isinstance(payload, str):
            raw_segments = [segment for segment in payload.splitlines() if segment.strip()]
        elif isinstance(payload, list):
            raw_segments = payload
        else:
            warnings.append(f"Unsupported transcript payload for {role_key}: {type(payload).__name__}")
            continue

        for segment in raw_segments:
            parsed_segment: List[Any] = []
            if isinstance(segment, str):
                try:
                    parsed_segment = json.loads(segment)
                except json.JSONDecodeError as exc:
                    warnings.append(f"Malformed JSON for {role_key}: {exc}")
                    continue
            elif isinstance(segment, list):
                parsed_segment = segment
            elif isinstance(segment, dict):
                parsed_segment = [segment]
            else:
                warnings.append(f"Unsupported transcript entry type for {role_key}: {type(segment).__name__}")
                continue

            parts = []
            for entry in parsed_segment:
                formatted = _format_structured_entry(entry)
                if formatted:
                    parts.append(formatted)
            if not parts:
                continue
            messages.append(Message(role=adapted_role, text=" ".join(parts).strip()))
    return messages, warnings


def normalize_record(raw: Dict[str, Any], redact_pii: bool = False) -> Conversation:
    call_id = _resolve_call_id(raw) or "unknown"
    structured_transcript = raw.get("transcript")
    if isinstance(structured_transcript, dict):
        messages, warnings = _parse_structured_transcript(structured_transcript)
        channel = "chat"
    else:
        transcript = raw.get("call_transcript", "") or ""
        messages, warnings = parse_transcript(transcript)
        channel = "call"
    metadata = {
        k: v
        for k, v in raw.items()
        if k not in {"call_transcript", "transcript"}
    }

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
        channel=channel,
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
