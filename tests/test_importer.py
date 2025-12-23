import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.importer import parse_transcript, normalize_record


def test_parse_transcript_basic():
    transcript = "[00:00:01] Agent: Hello.\n[00:00:02] Customer: Hi there."
    messages, warnings = parse_transcript(transcript)
    assert len(messages) == 2
    assert messages[0].role == "agent"
    assert messages[1].role == "customer"
    assert warnings == []


def test_parse_transcript_unparsed_line_appended():
    transcript = "[00:00:01] Agent: Hello.\nThis is a continuation"
    messages, warnings = parse_transcript(transcript)
    assert len(messages) == 1
    assert "continuation" in messages[0].text
    assert any("Unparsed line appended" in w for w in warnings)


def test_normalize_record_low_signal():
    raw = {"call_id": "c1", "call_transcript": "[00:00:01] Customer: Hi"}
    convo = normalize_record(raw)
    assert convo.metadata["is_low_signal"] is True
    assert convo.metadata["num_messages"] == 1
