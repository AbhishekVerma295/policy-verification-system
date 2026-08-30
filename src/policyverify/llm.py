"""
llm.py - the one place a language model is called.

Everything else in the system asks this module for text and knows nothing
about Ollama, Qwen, or HTTP. That is deliberate and it is the only abstraction
in the project that earns its keep: swapping the model - to a bigger Qwen, or
to a hosted API later - should be a config change, not a hunt through the
codebase for scattered client calls.

The interface is deliberately one method. `generate(prompt) -> str`. No
streaming, no chat history, no tools. Anything more would be abstraction built
for a future that may not arrive.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from policyverify.config import Config, get_config


class LLMError(RuntimeError):
    """The model could not be reached or refused to answer."""


@runtime_checkable
class LLMBackend(Protocol):
    """What the rest of the system needs from a language model."""

    def generate(self, prompt: str, *, json_mode: bool = True) -> str:
        """Return the model's reply as raw text."""
        ...


class OllamaBackend:
    """Runs a model locally through Ollama."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

    @property
    def model(self) -> str:
        return self.config.llm.model

    def generate(self, prompt: str, *, json_mode: bool = True) -> str:
        import ollama

        cfg = self.config.llm
        kwargs = {
            "model": cfg.model,
            "prompt": prompt,
            "options": {
                "temperature": cfg.temperature,
                "num_ctx": cfg.num_ctx,
            },
        }
        if json_mode:
            kwargs["format"] = "json"
        if cfg.disable_thinking:
            # See LLMConfig.disable_thinking - without this, Qwen3's reply
            # goes to the `thinking` field and `response` is empty.
            kwargs["think"] = False

        try:
            response = ollama.generate(**kwargs)
        except Exception as exc:  # connection refused, model not pulled, ...
            raise LLMError(
                f"could not reach Ollama for model {cfg.model!r}: {exc}\n"
                f"Is Ollama running, and have you pulled the model?\n"
                f"    ollama pull {cfg.model}"
            ) from exc

        text = response.get("response") or ""
        if not text.strip():
            # Almost always the thinking-mode trap described in config.py.
            thinking = response.get("thinking") or ""
            hint = (
                "\nThe model produced only 'thinking' output. Set "
                "llm.disable_thinking: true in config.yaml."
                if thinking.strip()
                else ""
            )
            raise LLMError(f"model {cfg.model!r} returned an empty response.{hint}")
        return text


def get_llm(config: Config | None = None) -> LLMBackend:
    """Build the configured backend.

    The single place a backend name turns into an object. Adding a hosted
    backend later means one more branch here and nothing else.
    """
    config = config or get_config()
    backend = config.llm.backend.lower()
    if backend == "ollama":
        return OllamaBackend(config)
    raise LLMError(
        f"unknown llm backend {config.llm.backend!r} - expected 'ollama'"
    )
