import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from understanding_agent.agent_working import (
    UnderstandingOutput, SingleAgentAnalysisSystem, parse_output,
)


def payload(references=None):
    return {section: [{"id": section + "_001", "claim": "A test observation",
                       "status": "computed", "dataframe_ids": ["data"],
                       "columns": ["x"], "evidence_ids": references or [],
                       "downstream_implication": "Review before modeling"}]
            for section in UnderstandingOutput.model_fields}


def test_evidence_toggle_and_bad_references():
    content = json.dumps(payload())
    parse_output(content, {}, False)
    with pytest.raises(ValueError, match="evidence required"):
        parse_output(content, {}, True)
    with pytest.raises(ValueError, match="unknown or failed"):
        parse_output(json.dumps(payload(["made_up"])), {}, False)


def test_unresolved_without_evidence_and_missing_sections():
    report = payload()
    for records in report.values():
        records[0]["status"] = "unresolved"
    parse_output(json.dumps(report), {}, True)
    del report["row_grain"]
    with pytest.raises(ValueError):
        parse_output(json.dumps(report), {}, False)


def test_duplicate_claim_ids_rejected():
    report = payload()
    report["row_grain"][0]["id"] = report["dataset_profile"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_output(json.dumps(report), {}, False)


class ScriptedModel:
    def __init__(self, output, call_tool=False):
        self.output, self.call_tool = output, call_tool
        self.prompts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        if self.call_tool:
            self.call_tool = False
            return AIMessage(content="", tool_calls=[{"name": "list_dfs_tool", "args": {}, "id": "ev1"}])
        return AIMessage(content=self.output)


def test_graph_evidence_and_structured_output():
    model = ScriptedModel(json.dumps(payload(["ev1"])), call_tool=True)
    agent = SingleAgentAnalysisSystem({"data": {"DataFrame": pd.DataFrame({"x": [1]})}},
                                     llm=model, require_evidence=True)
    result = agent.analyze()
    assert result["output_valid"]
    assert result["evidence"]["ev1"]["ok"]
    assert "MUST cite" in model.prompts[0]


def test_graph_invalid_output_preserved():
    agent = SingleAgentAnalysisSystem({"data": {"DataFrame": pd.DataFrame({"x": [1]})}},
                                     llm=ScriptedModel("not json"), require_evidence=False)
    result = agent.analyze()
    assert not result["output_valid"]
    assert result["raw_output"] == "not json"
    assert result["validation_errors"]
