from understanding_agent.executor import SubprocessExecutor


def test_real_analysis_and_fresh_state(tmp_path):
    (tmp_path / "data.csv").write_text("id,value\n1,2\n2,\n")
    (tmp_path / "columns.csv").write_text("column,description\nid,Identifier\nvalue,Value\n")
    executor = SubprocessExecutor(tmp_path)
    executor.check()
    result = executor.run("import pandas as pd; d = pd.read_csv(DATA_PATH); c = pd.read_csv(COLUMNS_PATH); print(d.shape, c.shape, d['value'].isna().sum()); x = 3")
    assert result["ok"] and "(2, 2) (2, 2) 1" in result["output"]
    assert not executor.run("print(x)")["ok"]


def test_timeout_and_output_limit(tmp_path):
    executor = SubprocessExecutor(tmp_path, timeout=2)
    result = executor.run("while True: pass")
    assert not result["ok"] and "timed out" in result["error"]
    result = executor.run("print('x' * 100000)")
    assert result["ok"] and result["truncated"] and len(result["output"]) == 16000


def test_compatible_model():
    from understanding_agent.agent import make_model
    model = make_model("test", "openai-compatible", "http://localhost:8000/v1")
    assert str(model.openai_api_base) == "http://localhost:8000/v1"


def test_preflight_no_run(tmp_path, monkeypatch, capsys):
    from understanding_agent.cli import main
    data = tmp_path / "test.csv"
    data.write_text("a\n1\n")
    monkeypatch.setattr("sys.argv", ["understand", "--data", str(data), "--columns", str(data),
                                   "--output", str(tmp_path / "runs"), "--check"])
    main()
    assert "executor ready" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
