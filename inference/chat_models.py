"""Tool-calling model clients for the existing ReAct agent."""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

def local_model(name: str, base_url: str):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=name, base_url=base_url, temperature=0, num_ctx=16384,
                      client_kwargs={"timeout": 120})


def make_model(name: str = "Qwen/Qwen3-0.6B", provider: str = "transformer", base_url: str | None = None):
    if provider == "transformer":
        from .transformers_backend import TransformersChatModel
        return TransformersChatModel(model_name=name)
    if provider == "ollama":
        return local_model(name, base_url)
    if provider == "openai-compatible":
        import os
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, base_url=base_url,
                          api_key=os.environ.get("DS_MODEL_API_KEY", "local"),
                          temperature=0, timeout=120, max_retries=1)
    raise ValueError(f"Unknown model provider: {provider}")


