"""Backend-neutral contract for future research rollout generation.

The tool-calling ReAct agent uses chat_models instead of this text-only API.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512


class GenerationBackend(ABC):
    @abstractmethod
    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[str]:
        """Return one continuation per prompt, in input order."""
        raise NotImplementedError
