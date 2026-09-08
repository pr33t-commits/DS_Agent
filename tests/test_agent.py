import json
import os
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from understanding_agent.agent import build_agent, validate_evidence
from understanding_agent.executor import DockerExecutor
from understanding_agent.schemas import UnderstandingReport


REPORT = {
    "summary": "Two customer records.", "row_count": 2, "column_count": 1,
    "proposed_row_grain": "One customer per row (hypothesis).",
    "glossary": [{"column": "id", "description": "Customer identifier", "source": "dictionary"}],
    "findings": [{"observation": "Two distinct IDs.", "evidence_ids": ["python_001"], "severity": "info"}],
    "business_questions": ["Is an ID unique across time?"], "limitations": ["No business brief."]
}


class ScriptedModel(BaseChatModel):
    step: int = 0

    @property
    def _llm_type(self):
        return "scripted-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.step == 0:
            call = {"name": "python_executor", "args": {"code": "print('two rows')"}, "id": "call_1"}
        else:
            call = {"name": "UnderstandingReport", "args": REPORT, "id": "call_2"}
        self.step += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


class StubExecutor:
    def run(self, code):
        return {"ok": True, "output": "two rows"}


def test_real_graph_tool_loop_and_report(tmp_path):
    graph, evidence = build_agent(StubExecutor(), tmp_path, ScriptedModel())
    state = graph.invoke({"messages": [{"role": "user", "content": "Understand the data"}]})
    report = state["structured_response"]
    validate_evidence(report, evidence)
    assert report.row_count == 2
    assert (tmp_path / "python_001.py").read_text() == "print('two rows')"
    assert json.loads((tmp_path / "python_001.json").read_text())["ok"]


@pytest.mark.parametrize("evidence", [{}, {"python_001": {"ok": False}}, {"python_002": {"ok": True}}])
def test_reject_unsupported_evidence(evidence):
    with pytest.raises(ValueError):
        validate_evidence(UnderstandingReport(**REPORT), evidence)


def test_execution_code_limit(tmp_path):
    assert not DockerExecutor(tmp_path).run("x" * 24001)["ok"]


def test_missing_docker_error_is_actionable(tmp_path):
    with patch("understanding_agent.executor.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Docker was not found on PATH"):
            DockerExecutor(tmp_path).check()


def test_local_model_configuration():
    from understanding_agent.agent import local_model
    model = local_model("test-model", "http://localhost:11434")
    assert model.temperature == 0
    assert model.num_ctx == 16384


def test_timeout_removes_container(tmp_path):
    import subprocess
    with patch("understanding_agent.executor.subprocess.Popen") as popen, patch("understanding_agent.executor.subprocess.run") as run:
        process = popen.return_value
        process.communicate.side_effect = [subprocess.TimeoutExpired("docker", 1), (None, None)]
        result = DockerExecutor(tmp_path, timeout=1).run("while True: pass")
        assert not result["ok"]
        process.kill.assert_called_once()
        assert run.call_args.args[0][:3] == ["docker", "rm", "-f"]
        command = popen.call_args.args[0]
        assert "--network=none" in command and "--read-only" in command


@pytest.mark.skipif(os.environ.get("DS_TEST_DOCKER") != "1", reason="Requires Docker and built image")
def test_docker_real_csv_and_error(tmp_path):
    (tmp_path / "data.csv").write_text("id,value\n1,2\n2,\n")
    (tmp_path / "columns.csv").write_text("column,description\nid,Identifier\nvalue,Measurement\n")
    executor = DockerExecutor(tmp_path)
    executor.check()
    result = executor.run("import pandas as pd; d = pd.read_csv('/inputs/data.csv'); print(d.shape); print(d['value'].isna().sum())")
    assert result["ok"] and "(2, 2)" in result["output"]
    assert not executor.run("raise ValueError('test')")["ok"]
    assert not executor.run("open('/inputs/data.csv', 'w')")["ok"]
