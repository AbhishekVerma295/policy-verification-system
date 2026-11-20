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

import os
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from policyverify.config import PROJECT_ROOT, Config, get_config


class LLMError(RuntimeError):
    """The model could not be reached or refused to answer."""


@runtime_checkable
class LLMBackend(Protocol):
    """What the rest of the system needs from a language model."""

    def generate(
        self, prompt: str, *, json_mode: bool = True, schema: dict | None = None
    ) -> str:
        """Return the model's reply as raw text.

        `schema`, when given, is a JSON schema the reply must follow. It is
        stronger than `json_mode`: json_mode only promises "some JSON", and
        measured on this model that let it invent its own field names, while
        a schema fixes the field names and types.
        """
        ...


class OllamaBackend:
    """Runs a model locally through Ollama."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

    @property
    def model(self) -> str:
        return self.config.llm.model

    def generate(
        self, prompt: str, *, json_mode: bool = True, schema: dict | None = None
    ) -> str:
        import ollama

        cfg = self.config.llm
        kwargs = {
            "model": cfg.model,
            "prompt": prompt,
            "keep_alive": cfg.keep_alive,
            "options": {
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "num_ctx": cfg.num_ctx,
                # A hard ceiling on reply length - see LLMConfig.max_tokens.
                "num_predict": cfg.max_tokens,
            },
        }
        if schema is not None:
            # Ollama "structured outputs": decoding itself is constrained to
            # the schema, so the field names and types cannot drift.
            kwargs["format"] = schema
        elif json_mode:
            kwargs["format"] = "json"
        if cfg.disable_thinking:
            # See LLMConfig.disable_thinking - without this, Qwen3's reply
            # goes to the `thinking` field and `response` is empty.
            kwargs["think"] = False

        try:
            # A client with a timeout, so a request that never finishes is
            # an error the caller sees rather than a hang.
            client = ollama.Client(timeout=cfg.timeout_seconds)
            response = client.generate(**kwargs)
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
        log_exchange(prompt, text, cfg.model)
        return text


PROMPT_LOG_ENV = "POLICYVERIFY_LOG_PROMPTS"


def log_exchange(prompt: str, reply: str, model: str) -> Path | None:
    """Save one prompt and the model's reply, when prompt logging is on.

    Set POLICYVERIFY_LOG_PROMPTS=1 (or to a directory) and every call to the
    model is written to data/logs/prompts/ as a plain text file: exactly what
    the model was shown - instructions, passages, summaries - and exactly
    what it said back. This is the way to answer "what did the LLM actually
    receive?" for any question, without guessing from the code.
    """
    target = os.environ.get(PROMPT_LOG_ENV, "").strip()
    if not target or target == "0":
        return None
    folder = (
        Path(target)
        if target not in ("1", "true", "yes")
        else PROJECT_ROOT / "data" / "logs" / "prompts"
    )
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = folder / f"{stamp}.txt"
    path.write_text(
        f"MODEL: {model}\n\n===== PROMPT =====\n{prompt}\n\n===== REPLY =====\n{reply}\n",
        encoding="utf-8",
    )
    return path


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
