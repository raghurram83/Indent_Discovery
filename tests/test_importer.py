import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.importer import normalize_record, parse_transcript, read_raw_records


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
    assert convo.metadata["num_messages"] == 0
    assert convo.messages_for_llm == []


def test_normalize_record_chat_transcript():
    raw = {
        "Conversation_id": "conv-chat-1",
        "Actor first name": "Agent Smith",
        "start timestamp": "2025-11-19T14:45:40.042Z",
        "transcript": {
            "user": '[{"text":{"content":"Hi"}}]',
            "Agnt": '[{"text":{"content":"Hello, how can I help?"}}]',
        },
    }
    convo = normalize_record(raw)
    assert convo.channel == "chat"
    assert len(convo.messages) == 2
    assert convo.messages[0].role == "customer"
    assert convo.messages[0].text == "Hi"
    assert convo.messages[1].role == "agent"
    assert "transcript" not in convo.metadata


def test_read_raw_records_flattens_conversations(tmp_path):
    sample = {
        "run": "run_1",
        "generated_at": "2026-01-18T14:40:16.784Z",
        "conversations": [
            {"Conversation_id": "c1", "Actor first name": "A"},
            {"Conversation_id": "c2", "Actor first name": "B"},
        ],
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(sample))
    rows = read_raw_records(path)
    assert len(rows) == 2
    assert rows[0]["Conversation_id"] == "c1"
    assert rows[0]["run"] == "run_1"


def test_read_raw_records_plain_text(tmp_path):
    path = tmp_path / "call_123.txt"
    path.write_text("[00:00:01] Agent: Hello\\n[00:00:02] Customer: Hi")
    rows = read_raw_records(path)
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_123"
    assert "Customer: Hi" in rows[0]["call_transcript"]


def test_read_raw_records_batch_txt(tmp_path):
    content = """Batch Transcription Results
Generated: 2026-03-13T12:18:03.945820
Total recordings: 2
Model: gemini-2.5-flash

================================================================================
Recording #2: abc123.mp3
URL: https://example.com/abc123.mp3
Timestamp: 2026-03-13T12:18:09.068565
================================================================================

[ 00:00:00] Agent: Hello.
[ 00:00:02] Customer: Hi.

================================================================================
Recording #3: def456.mp3
URL: https://example.com/def456.mp3
Timestamp: 2026-03-13T12:18:10.000000
================================================================================

[ 00:00:00 ] Agent: Hola
"""
    path = tmp_path / "batch.txt"
    path.write_text(content)
    rows = read_raw_records(path)
    assert len(rows) == 2
    assert rows[0]["call_id"] == "abc123"
    assert "Hello." in rows[0]["call_transcript"]
    assert rows[0]["recording_url"].endswith("abc123.mp3")
    assert rows[1]["call_id"] == "def456"
    assert "Hola" in rows[1]["call_transcript"]
