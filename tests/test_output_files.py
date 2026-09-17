import json
from understanding_agent.output_files import save_readable_output


def test_invalid_report_prose_is_saved_verbatim(tmp_path):
    path = tmp_path / "raw.txt"
    text = "Findings\n\nRevenue is uncertain.\n• Ask about units."
    save_readable_output({"output_valid": False, "raw_output": text}, path)
    assert path.read_text(encoding="utf-8") == text


def test_schema_invalid_json_is_readable(tmp_path):
    path = tmp_path / "raw.txt"
    save_readable_output({"output_valid": False, "raw_output": '{"unexpected":"value"}'}, path)
    assert json.loads(path.read_text()) == {"unexpected": "value"}
    assert "\n" in path.read_text()


def test_text_blocks_and_message_fallback(tmp_path):
    path = tmp_path / "raw.txt"
    save_readable_output({"messages": [{"content": [{"type": "text", "text": "First"},
                                                   {"type": "text", "text": "Second"}]}]}, path)
    assert path.read_text() == "First\n\nSecond"
