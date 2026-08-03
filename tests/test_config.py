"""Env-driven, local-first model configuration (no network)."""

from __future__ import annotations

import pytest

from dlens.config import DEFAULT_MODEL, DEFAULT_PROVIDER, ModelSettings, build_model_from_env


def _clear(monkeypatch):
    for var in (
        "DLENS_MODEL_PROVIDER", "DLENS_MODEL", "DLENS_MODEL_PORT",
        "DLENS_MODEL_BASE_URL", "DLENS_MODEL_API_KEY", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_local_ollama_qwen(monkeypatch):
    _clear(monkeypatch)
    s = ModelSettings.from_env()
    assert s.provider == DEFAULT_PROVIDER == "ollama"
    assert s.model == DEFAULT_MODEL == "qwen3:8b"
    model = build_model_from_env()
    # OllamaModel subclasses OpenAIChatModel; tag is qwen3:8b.
    assert "qwen3:8b" in str(getattr(model, "model_name", ""))


def test_env_overrides_provider_and_model(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("DLENS_MODEL", "Qwen/Qwen3-8B")
    s = ModelSettings.from_env()
    assert s.provider == "vllm" and s.model == "Qwen/Qwen3-8B"
    build_model_from_env()  # constructs without error / network


def test_openai_compatible_requires_base_url(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DLENS_MODEL_API_KEY", "shared-key")
    with pytest.raises(ValueError, match="DLENS_MODEL_BASE_URL"):
        build_model_from_env()
    # With a base URL it builds (shared-key / gateway path).
    monkeypatch.setenv("DLENS_MODEL_BASE_URL", "https://gateway.example/v1")
    build_model_from_env()


def test_unknown_provider_raises(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "nonsense")
    with pytest.raises(ValueError, match="Unknown DLENS_MODEL_PROVIDER"):
        build_model_from_env()


def test_describe_is_key_safe(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DLENS_MODEL_API_KEY", "super-secret")
    monkeypatch.setenv("DLENS_MODEL_BASE_URL", "https://gateway.example/v1")
    desc = ModelSettings.from_env().describe()
    assert "super-secret" not in desc
    assert "api_key=set" in desc


def test_openai_provider_defaults_to_gpt_4o_mini(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = ModelSettings.from_env()
    assert s.provider == "openai"
    assert s.model == "gpt-4o-mini"  # provider-specific default, not the ollama tag
    assert s.api_key == "sk-test"
    build_model_from_env()  # constructs without a network call
