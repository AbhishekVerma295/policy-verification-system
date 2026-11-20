"""
Tests for per-machine settings and the prompt log.

config.local.yaml lets one machine run a different model without editing the
committed config.yaml; the prompt log shows exactly what the model received.
"""

from __future__ import annotations

from policyverify import config as config_module
from policyverify.config import load_config
from policyverify.llm import PROMPT_LOG_ENV, log_exchange


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_local_config_overrides_only_what_it_names(tmp_path, monkeypatch):
    base = _write(tmp_path / "config.yaml", "llm:\n  model: qwen3:4b\n  max_tokens: 1500\n")
    _write(tmp_path / "config.local.yaml", "llm:\n  model: qwen2.5:3b\n")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", base)
    monkeypatch.delenv(config_module.LLM_MODEL_ENV, raising=False)

    config = load_config()
    assert config.llm.model == "qwen2.5:3b"
    assert config.llm.max_tokens == 1500, "unnamed settings keep config.yaml's value"


def test_an_explicit_config_path_ignores_the_local_file(tmp_path, monkeypatch):
    base = _write(tmp_path / "config.yaml", "llm:\n  model: qwen3:4b\n")
    _write(tmp_path / "config.local.yaml", "llm:\n  model: qwen2.5:3b\n")
    monkeypatch.delenv(config_module.LLM_MODEL_ENV, raising=False)
    assert load_config(base).llm.model == "qwen3:4b"


def test_the_environment_variable_still_wins(tmp_path, monkeypatch):
    base = _write(tmp_path / "config.yaml", "llm:\n  model: qwen3:4b\n")
    _write(tmp_path / "config.local.yaml", "llm:\n  model: qwen2.5:3b\n")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", base)
    monkeypatch.setenv(config_module.LLM_MODEL_ENV, "llama3:8b")
    assert load_config().llm.model == "llama3:8b"


def test_near_zero_temperature_and_a_fixed_seed_by_default():
    llm = config_module.LLMConfig()
    assert 0 < llm.temperature <= 0.001
    assert isinstance(llm.seed, int)


def test_prompt_log_is_off_unless_asked_for(tmp_path, monkeypatch):
    monkeypatch.delenv(PROMPT_LOG_ENV, raising=False)
    assert log_exchange("prompt", "reply", "m") is None


def test_prompt_log_records_the_exact_prompt_and_reply(tmp_path, monkeypatch):
    monkeypatch.setenv(PROMPT_LOG_ENV, str(tmp_path))
    path = log_exchange("PASSAGES: rule text\nQUESTION: q", '{"claims": []}', "qwen2.5:3b")
    text = path.read_text(encoding="utf-8")
    assert "PASSAGES: rule text\nQUESTION: q" in text
    assert '{"claims": []}' in text
    assert "qwen2.5:3b" in text
