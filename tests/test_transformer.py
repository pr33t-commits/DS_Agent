import json

import pytest
from pydantic import Field

from inference.chat_models import make_model
from inference.transformers_backend import TransformersChatModel, parse_response
from understanding_agent.agent import build_agent, validate_evidence
from test_agent import REPORT, StubExecutor


def test_default_factory_is_lazy_transformer():
    model = make_model()
    assert isinstance(model, TransformersChatModel)
    assert model.model_name == "Qwen/Qwen3-0.6B"
    assert model._model is None


def test_parse_multiple_calls_and_ignore_reasoning():
    text = '<think><tool_call>not executable</tool_call></think>Done'
    text += '<tool_call>{"name":"one","arguments":{"x":1}}</tool_call>'
    text += '<tool_call>{"name":"two","arguments":"{}"}</tool_call>'
    result = parse_response(text)
    assert result.content == "Done"
    assert [c["name"] for c in result.tool_calls] == ["one", "two"]
    assert result.tool_calls[0]["id"] != result.tool_calls[1]["id"]


@pytest.mark.parametrize("text", [
    '<tool_call>{', '<tool_call>bad</tool_call>',
    '<tool_call>{"name":"x","arguments":[]}</tool_call>', '<think>unfinished',
])
def test_reject_bad_or_truncated_calls(text):
    with pytest.raises(ValueError):
        parse_response(text)


class ScriptedTransformer(TransformersChatModel):
    seen_tools: list = Field(default_factory=list)
    step: int = 0

    def _completion(self, messages, tools):
        self.seen_tools.extend(t["function"]["name"] for t in tools)
        if self.step == 0:
            call = {"name": "python_executor", "arguments": {"code": "print('two rows')"}}
        else:
            assert any(m.type == "tool" for m in messages)
            call = {"name": "UnderstandingReport", "arguments": REPORT}
        self.step += 1
        return '<tool_call>' + json.dumps(call) + '</tool_call>'


def test_transformer_real_graph_tool_and_structured_report(tmp_path):
    model = ScriptedTransformer()
    graph, evidence = build_agent(StubExecutor(), tmp_path, model)
    state = graph.invoke({"messages": [{"role": "user", "content": "Understand"}]})
    validate_evidence(state["structured_response"], evidence)
    assert state["structured_response"].row_count == 2
    assert "python_executor" in model.seen_tools
    assert "UnderstandingReport" in model.seen_tools


def test_cli_default_transformer(tmp_path, monkeypatch, capsys):
    from understanding_agent.cli import main
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n")
    monkeypatch.setattr("sys.argv", ["understand", "--check", "--data", str(data), "--columns", str(data)])
    main()
    assert "transformer (Qwen/Qwen3-0.6B)" in capsys.readouterr().out
