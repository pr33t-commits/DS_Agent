import argparse
import hashlib
import json
import shutil
import sys
import platform
import re
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
import pandas as pd, numpy as np

from .agent import SYSTEM_PROMPT, build_agent, make_model, validate_evidence
from .executor import DockerExecutor, SubprocessExecutor
from .agent_working import SingleAgentAnalysisSystem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "DataCoSupplyChainDataset.csv"
DEFAULT_COLUMNS = PROJECT_ROOT / "data" / "DescriptionDataCoSupplyChain.csv"


def read_input_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, preserving UTF-8 as the default with a legacy fallback.

    The bundled DataCo dataset uses a Windows/Latin-1-compatible encoding, while
    the column dictionary is UTF-8.  Trying UTF-8 first keeps that standard for
    user-provided data and only uses the fallback when it is necessary.
    """
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def main():
    parser = argparse.ArgumentParser(description="Understand a CSV and its column dictionary with a local ReAct agent.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--columns", type=Path, default=DEFAULT_COLUMNS)
    parser.add_argument("--model", help="Hugging Face model ID/local checkpoint for transformer; server model name otherwise", default = "Qwen/Qwen3-0.6B")
    parser.add_argument("--provider", choices=["transformer", "ollama", "openai-compatible"], default="transformer")
    parser.add_argument("--base-url", help="Defaults to localhost:11434 for Ollama or localhost:8000/v1 for compatible servers")
    parser.add_argument("--executor", choices=["subprocess", "docker"], default="subprocess")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess analysis")
    parser.add_argument("--execution-timeout", type=int, default=45)
    parser.add_argument("--check", action="store_true", help="Check files and executor without invoking a model or creating a run")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--experiment-id",
                        help="Folder name for this experiment (default: generated experiment ID)")
    parser.add_argument("--image", default="ds-understanding-python:local")
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=1,
                        help="Number of independent agent analyses to run (default: 1)")
    parser.add_argument("--context", default="No business context provided.")
    args = parser.parse_args()
    args.model = args.model or ("Qwen/Qwen3-0.6B" if args.provider == "transformer" else "Qwen3-0.6B")
    if args.provider == "transformer" and args.base_url:
        parser.error("--base-url applies only to server providers")
    args.base_url = args.base_url or {"ollama": "http://localhost:11434", "openai-compatible": "http://localhost:8000/v1"}.get(args.provider)
    for path in (args.data, args.columns):
        if not path.is_file():
            parser.error(f"CSV file not found: {path}")
    if not 1 <= args.max_calls <= 50:
        parser.error("--max-calls must be between 1 and 50")
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.execution_timeout < 1:
        parser.error("--execution-timeout must be positive")
    experiment_id = args.experiment_id or f"experiment_{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
        parser.error("--experiment-id may contain only letters, numbers, dots, underscores, and hyphens")
    def get_executor(inputs):
        if args.executor == "docker":
            return DockerExecutor(inputs, args.image, args.execution_timeout)
        return SubprocessExecutor(inputs, args.python, args.execution_timeout)
    try:
        get_executor(args.data.parent).check()
    except Exception as exc:
        parser.exit(1, f"Preflight failed: {exc}\n")
    if args.check:
        print(f"Inputs and {args.executor} executor ready. Model not checked: {args.provider} ({args.model})")
        return
    if args.executor == "subprocess":
        print("Python execution uses this account's permissions (not sandboxed).", flush=True)
    run_dir = args.output.resolve() / experiment_id
    if run_dir.exists():
        parser.error(f"Experiment output already exists: {run_dir}. Choose a new --experiment-id.")
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    manifest = {"model": args.model, "base_url": args.base_url, "image": args.image,
                "provider": args.provider, "executor": args.executor, "python": args.python,
                "execution_timeout": args.execution_timeout, "platform": platform.platform(),
                "packages": {p: version(p) for p in ("langchain", "langgraph", "pydantic")},
                "experiment_id": experiment_id, "max_calls": args.max_calls, "rollouts": args.rollouts,
                "context": args.context, "inputs": {}}
    df_dict = {}
    for source, name, description in ((args.data, "sales_delivery_data.csv", "Sales and Delivery Data"), (args.columns, "data_dictionary.csv", "Data Dictionary")):
        target = inputs / name
        shutil.copyfile(source, target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["inputs"][name] = {"source": str(source.resolve()), "sha256": digest}
        dict_key = name.split(".")[0]
        df_dict[dict_key] = {}
        df_dict[dict_key]['DataFrame'] = read_input_csv(source)
        df_dict[dict_key]['Description'] = description
        
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    try:
        executor = get_executor(inputs)
        # agent, evidence = build_agent(executor, run_dir, make_model(args.model, args.provider, args.base_url), args.max_calls)
        shared_llm = make_model(args.model, args.provider, args.base_url)
        rollout_dir = run_dir / "rollouts"
        rollout_dir.mkdir()
        rollout_summary = []

        def json_default(value):
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            return str(value)

        # with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:

            # for state in agent.stream({"messages": [{"role": "user", "content":
            #         "Understand both supplied CSVs. Business context: " + args.context}]},
            #         config={"recursion_limit": 2 * args.max_calls + 10}, stream_mode="values"):
            #     final = state
            #     snapshot = {"messages": [m.model_dump(mode="json") for m in state["messages"]]}
            #     trace.write(json.dumps(snapshot, default=str) + "\n")
            #     trace.flush()
        for rollout_number in range(1, args.rollouts + 1):
            print(f"Starting rollout {rollout_number}/{args.rollouts}", flush=True)
            # A fresh system means no messages or generated DataFrames leak
            # between rollouts; the shared client keeps model weights in memory.
            agent = SingleAgentAnalysisSystem(dataframes=df_dict, model=args.model,
                                              llm=shared_llm)
            output_path = rollout_dir / f"final_state_{rollout_number:03d}.json"
            try:
                final_state = agent.analyze()
                output_path.write_text(
                    json.dumps(final_state, indent=2, default=json_default),
                    encoding="utf-8",
                )
                rollout_summary.append({"rollout": rollout_number, "status": "ok",
                                        "final_state": str(output_path.relative_to(run_dir))})
                print(f"Saved rollout {rollout_number}: {output_path}", flush=True)
            except Exception as exc:
                error_path = rollout_dir / f"error_{rollout_number:03d}.json"
                error_path.write_text(
                    json.dumps({"type": type(exc).__name__, "error": str(exc)}, indent=2),
                    encoding="utf-8",
                )
                rollout_summary.append({"rollout": rollout_number, "status": "error",
                                        "error": str(error_path.relative_to(run_dir))})
                print(f"Rollout {rollout_number} failed: {exc}", file=sys.stderr, flush=True)

        (run_dir / "rollout_summary.json").write_text(
            json.dumps(rollout_summary, indent=2), encoding="utf-8"
        )
        failed = [item for item in rollout_summary if item["status"] == "error"]
        if failed:
            parser.exit(1, f"{len(failed)} rollout(s) failed. See {run_dir / 'rollout_summary.json'}\n")
        print(f"Saved {args.rollouts} rollout final state(s): {rollout_dir}")
        # print(final['messages'])
        # report = final["structured_response"]
        # # validate_evidence(report, evidence)
        # (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        # lines = ["# Dataset understanding", "", report.summary, "",
        #          f"Rows: {report.row_count}; columns: {report.column_count}", "",
        #          "Proposed row grain: " + report.proposed_row_grain, "", "## Glossary", ""]
        # lines += [f"- **{c.column}** ({c.source}): {c.description}" for c in report.glossary]
        # lines += ["", "## Findings", ""] + [f"- [{f.severity}] {f.observation} ({', '.join(f.evidence_ids)})" for f in report.findings]
        # lines += ["", "## Business questions", ""] + ["- " + q for q in report.business_questions]
        # lines += ["", "## Limitations", ""] + ["- " + q for q in report.limitations]
        # (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        # print(f"Report saved: {run_dir / 'report.md'}")
        
    except Exception as exc:
        (run_dir / "error.json").write_text(json.dumps({"type": type(exc).__name__, "error": str(exc)}, indent=2), encoding="utf-8")
        parser.exit(1, f"Run failed: {exc}\nPartial trace: {run_dir}\n")


if __name__ == "__main__":
    main()
