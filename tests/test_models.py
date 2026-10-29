"""Offline contract tests for OpenAI-compatible local model wrappers."""

from __future__ import annotations

from dlens.agents._models import LiteLLMModel, VLLMModel


def test_vllm_model_uses_local_default_configuration():
    model = VLLMModel()

    assert model.model_name == "gpt-oss:20b"
    assert str(model.provider.base_url) == "http://localhost:8000/v1/"


def test_vllm_model_accepts_custom_model_and_port():
    model = VLLMModel(model_name="Qwen/Qwen3-8B", port="9001")

    assert model.model_name == "Qwen/Qwen3-8B"
    assert str(model.provider.base_url) == "http://localhost:9001/v1/"


def test_litellm_model_uses_local_default_configuration():
    model = LiteLLMModel()

    assert model.model_name == "local-ollama"
    assert str(model.provider.base_url) == "http://127.0.0.1:4000/v1/"


def test_litellm_model_accepts_custom_model_and_port():
    model = LiteLLMModel(model_name="local-qwen", port="4100")

    assert model.model_name == "local-qwen"
    assert str(model.provider.base_url) == "http://127.0.0.1:4100/v1/"
