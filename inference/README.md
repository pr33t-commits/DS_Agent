# Inference layer

`chat_models.py` selects the default `transformer` provider, or optional Ollama
and OpenAI-compatible clients. `transformers_backend.py` implements local agent
inference using Transformers, including Qwen-style tool-call parsing. Install
with `python -m pip install -e '.[transformer]'`. Model weights load lazily.
The default checkpoint is `Qwen/Qwen3-0.6B`; other tool syntaxes need a parser.

`generation.py` defines the separate text rollout contract for future research.
Training/rollout backends are not yet implemented. The compatible client can
connect to an independently running vLLM server.
