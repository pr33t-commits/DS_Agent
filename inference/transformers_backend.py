"""Local Transformers chat adapter for Qwen-compatible tool-call templates."""
import json
import re
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


def parse_response(text: str) -> AIMessage:
    # Never execute tool examples emitted inside a reasoning block.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Incomplete model reasoning; increase max_new_tokens.")
    calls = []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    for payload in re.findall(pattern, text, flags=re.DOTALL):
        try:
            call = json.loads(payload)
            args = call["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(call["name"], str) or not isinstance(args, dict):
                raise ValueError("Expected a tool name and argument object")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Malformed Transformers tool call") from exc
        calls.append({"name": call["name"], "args": args, "id": "call_" + uuid4().hex})
    content = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("Incomplete Transformers tool call; increase max_new_tokens.")
    return AIMessage(content=content, tool_calls=calls)


class TransformersChatModel(BaseChatModel):
    """Loads weights on first invocation; uses CUDA when available, otherwise CPU.

    Tool output must use Qwen's <tool_call> JSON format. Other model-specific
    tool syntaxes require their own parser. No remote custom code is enabled.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    max_new_tokens: int = 2048
    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self):
        return "transformer"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "max_new_tokens": self.max_new_tokens}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # LangChain requests 'any' for structured output. Templates describe
        # the available tools; generation does not enforce a tool choice.
        if tool_choice not in (None, "auto", "any", "required"):
            raise ValueError("This provider does not support forcing a named tool")
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _completion(self, messages, tools):
        # try:
        #     import torch
        #     from transformers import AutoModelForCausalLM, AutoTokenizer
        # except ImportError as exc:
        #     raise ImportError('Transformer provider requires: python -m pip install -e ".[transformer]"') from exc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
        
        for i, msg in enumerate(messages):
            print(i, type(msg), repr(msg))
        history = convert_to_openai_messages(messages)
        for message in history:
            for call in message.get("tool_calls", []):
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args)
        inputs = self._tokenizer.apply_chat_template(
            history, tools=tools or None, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=True,
        ).to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError("Custom stop sequences are not supported by this provider")
        message = parse_response(self._completion(messages, kwargs.get("tools", [])))
        return ChatResult(generations=[ChatGeneration(message=message)])
