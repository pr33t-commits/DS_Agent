# Dataset understanding agent

Python ReAct agent built on LangChain/LangGraph. Consumes a data CSV and a column
dictionary, executes generated Python and saves evidence-backed JSON/Markdown reports.

## Setup: Windows or Linux / Jupyter terminal

Python 3.11+ is required. From the checkout, create a virtual environment if needed:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in Linux. An existing suitable environment also works.

```bash
python -m pip install -e '.[dev]'
python -m understanding_agent.cli --check
```

The default `transformer` provider runs Hugging Face models in this Python
process. Install its dependencies with `python -m pip install -e '.[dev,transformer]'`.
It uses CUDA when available and otherwise CPU. First invocation downloads model
weights unless cached; `--check` does not load or download a model.
Server providers can still use the base installation without PyTorch.

Defaults: `data/DataCoSupplyChainDataset.csv`, `data/DescriptionDataCoSupplyChain.csv`
and `experiments/runs/`, resolved relative to the checkout regardless of working directory.
Override using `--data`, `--columns`, `--output`. Explicit relative paths resolve
from your terminal. Transfer the data along with the project when moving machines.
`--check` checks inputs and executor dependencies without creating a run or calling
the model. Use the installed `understand` command from a Jupyter terminal too.

## Model providers

`transformer` is the default, using `Qwen/Qwen3-0.6B` locally:

```bash
python -m understanding_agent.cli
python -m understanding_agent.cli --provider transformer --model Qwen/Qwen3-0.6B
```

`--model` accepts a Hugging Face model ID or local checkpoint directory. The
adapter uses the tokenizer's chat template and parses Qwen-style `<tool_call>`
JSON. Models with other tool-output formats need a corresponding parser.
Generation is greedy, thinking is disabled, and each response is capped at 2048
new tokens. Model quality and CPU speed depend on the checkpoint and hardware.
`--base-url` is only valid for server providers.

Ollama remains available at localhost:11434:

```bash
python -m understanding_agent.cli --provider ollama --model Qwen3-0.6B
```

For server providers, the model name must match a registered model exactly.

For an already running vLLM or another OpenAI-compatible server:

```bash
python -m understanding_agent.cli --provider openai-compatible --base-url http://localhost:8000/v1 --model Qwen3-0.6B
```

Configure model serving separately with the appropriate tool-call parser and chat
template. The agent requires tool calling and does not launch inference services.
Set `DS_MODEL_API_KEY` if authentication is required. The compatible client sends
requests only to your configured endpoint. Use localhost on the GPU VM or an SSH
tunnel from your laptop. A fine-tuned checkpoint can later use the same interface.

## Execution

`--executor subprocess` is the default on both Windows and Linux. Each call uses
a fresh Python process and temporary working directory. Predefined `DATA_PATH`
and `COLUMNS_PATH` variables point to staged input copies; use `pd.read_csv(DATA_PATH)`.
State is not preserved. Select another Python using `--python /path/to/python`.
Adjust the 45-second timeout using `--execution-timeout 120`.

Subprocess execution is **not a security sandbox**. Code runs with the account's
filesystem/network permissions, can modify files and has no hard CPU/memory cap.
Use a dedicated development environment. Output is bounded to 16,000 bytes and
normal timed-out process trees are terminated; this does not contain hostile code.

Docker remains an explicit optional backend where Docker Engine is available:

```bash
docker build -t ds-understanding-python:local sandbox
python -m understanding_agent.cli --executor docker
```

It provides read-only input mounts, disabled networking and resource limits. Do
not assume nested Docker is available inside rented containers. Full Linux VMs
can use Docker Engine without Docker Desktop. There is no automatic mode fallback.

## Runs and evaluation

Runs save input copies/hashes, configuration, package versions, prompt, scripts,
tool outputs, message traces and reports. Preflight errors create no run; model
failures retain partial artifacts. Set `--output` to persistent storage on rented
machines and copy runs before destroying an instance. Traces may contain data;

To run independent analyses of the same inputs, use `--rollouts`:

```bash
python -m understanding_agent.cli --experiment-id data-quality-v1 --rollouts 5 --output runs
```

The model is loaded once and reused, but every rollout has fresh agent state.
Its final state is saved as `rollouts/final_state_001.json` (and so on) under a
experiment directory. If no `--experiment-id` is supplied, one is generated.
`rollout_summary.json` lists successful and failed
rollouts; a failed rollout writes its error to `rollouts/error_###.json` without
discarding completed rollout outputs.
the project does not enable hosted tracing (external environment settings still apply).

```bash
python -m pytest -q
```

Tests exercise the real graph with a scripted model and actual Python subprocesses.
Docker integration requires `DS_TEST_DOCKER=1` and the image. CI defines Windows
and Linux jobs without GPU/model downloads. Actual model quality is not established
by these tests. Evidence validation checks references, not truth or full profiling
coverage. Context history is not automatically summarized. Correct and review
traces before using them as training data; post-training is not implemented yet.

References: [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents),
[Vast.ai environment](https://docs.vast.ai/guides/instances/docker-environment).

## Project layout

```text
DS_Agent/
├── configs/                 # Future experiment configuration files
├── data/                    # Raw CSVs and shared dataset locations
│   └── datasets.py
├── training/                # Planned SFT / DPO / GRPO implementations
├── inference/
│   ├── chat_models.py       # Existing tool-calling model clients
│   └── generation.py        # Shared research generation interface
├── evaluation/
│   └── evidence.py          # Report evidence validation
├── experiments/
│   └── runs/               # Existing and future run artifacts
├── understanding_agent/     # ReAct orchestration, CLI, schemas, execution
├── tests/
└── sandbox/                 # Optional Docker executor image
```

This layout follows the supplied post-training architecture while preserving
current agent behavior. Research training will use Transformers/PyTorch, with
Accelerate, PEFT and TRL as needed; vLLM is a separate rollout backend. Training
algorithms and research rollout backends are not implemented yet. Local agent
inference is implemented by `inference/transformers_backend.py`.
The existing Ollama and compatible-server clients remain available for the agent.
Each layer's README describes its current responsibilities and planned extensions.

Existing runs from both `runs/` and `understanding_agent/runs/` have moved into
`experiments/runs/`. Their contents and historical manifest paths are preserved.
The old `understanding_agent.agent` model/evidence imports remain compatibility
exports; new code should import from `inference.chat_models` and
`evaluation.evidence` directly. Reinstall the editable package after restructuring:

```bash
python -m pip install -e '.[dev]'
```
