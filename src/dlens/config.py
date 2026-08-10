# config.py
"""Environment-driven model selection. Model-agnostic and local-first.

Lets the local model be chosen via environment variables without code changes,
while reusing the model wrappers in ``dlens.agents._models`` (no parallel factory).
Defaults to a local Qwen model on Ollama (``qwen3:8b`` — ~5.2 GB, runs on a single
workstation GPU and supports tool calling).

Recognized variables (all optional):

    DLENS_MODEL_PROVIDER   ollama | vllm | llamacpp | openai | openai_compatible
                           (default: ollama)
    DLENS_MODEL            model name/tag                     (default: qwen3:8b)
    DLENS_MODEL_PORT       port for local servers             (provider default)
    DLENS_MODEL_BASE_URL   base URL for openai_compatible (e.g. a shared vLLM gateway / OpenRouter)
    DLENS_MODEL_API_KEY    API key for openai_compatible / openai (else OPENAI_API_KEY)

Recommended local models:
    qwen3:8b   (default, ~5.2 GB) — single-GPU friendly, tool calling.
    gpt-oss:20b (~14 GB)          — stronger reasoning when a larger GPU is available.
Smaller/larger Qwen tags (qwen3:4b ~2.5 GB ... qwen3:32b ~20 GB) trade VRAM for capability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # load a local .env if present (used for keys on the shared/API paths)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a declared dep; stay defensive
    pass

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from dlens.agents._models import LlamaCppModel, OllamaModel, OpenAIModel, VLLMModel

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3:8b"
_DEFAULT_PORT = {"ollama": "11434", "vllm": "8000", "llamacpp": "8080"}
# Default model per provider when DLENS_MODEL is not set (local tags vs a hosted name).
_DEFAULT_MODEL_BY_PROVIDER = {
    "ollama": "qwen3:8b",
    "vllm": "qwen3:8b",
    "llamacpp": "qwen3:8b",
    "openai": "gpt-4o-mini",
    "openai_compatible": "qwen3:8b",
}


@dataclass(frozen=True)
class ModelSettings:
    """Resolved local-model configuration."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    port: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "ModelSettings":
        provider = os.getenv("DLENS_MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        # Model default is provider-specific (e.g. gpt-4o-mini for openai), unless set.
        model = (os.getenv("DLENS_MODEL") or _DEFAULT_MODEL_BY_PROVIDER.get(provider, DEFAULT_MODEL)).strip()
        return cls(
            provider=provider,
            model=model,
            port=os.getenv("DLENS_MODEL_PORT") or None,
            base_url=os.getenv("DLENS_MODEL_BASE_URL") or None,
            api_key=os.getenv("DLENS_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
        )

    def build(self) -> Model:
        """Construct a pydantic-ai model, reusing the dlens model wrappers."""
        if self.provider == "ollama":
            return OllamaModel(model_name=self.model, port=self.port or _DEFAULT_PORT["ollama"])
        if self.provider == "vllm":
            return VLLMModel(model_name=self.model, port=self.port or _DEFAULT_PORT["vllm"])
        if self.provider == "llamacpp":
            return LlamaCppModel(model_name=self.model, port=self.port or _DEFAULT_PORT["llamacpp"])
        if self.provider == "openai":
            return OpenAIModel(model_name=self.model)
        if self.provider == "openai_compatible":
            # Shared key / remote OpenAI-compatible gateway (e.g. a shared vLLM
            # endpoint or OpenRouter). Switchable for later without code changes.
            if not self.base_url:
                raise ValueError(
                    "provider=openai_compatible requires DLENS_MODEL_BASE_URL "
                    "(and usually DLENS_MODEL_API_KEY)."
                )
            return OpenAIChatModel(
                self.model,
                provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key or "local"),
            )
        raise ValueError(
            f"Unknown DLENS_MODEL_PROVIDER={self.provider!r} "
            "(expected ollama | vllm | llamacpp | openai | openai_compatible)."
        )

    def describe(self) -> str:
        """One-line, key-safe summary for logs/demos."""
        key = "set" if self.api_key else "none"
        return (
            f"provider={self.provider} model={self.model} "
            f"port={self.port or _DEFAULT_PORT.get(self.provider, '-')} "
            f"base_url={self.base_url or '-'} api_key={key}"
        )


def build_model_from_env() -> Model:
    """Build a model from environment variables (defaults to local Ollama qwen3:8b)."""
    return ModelSettings.from_env().build()


# --------------------------------------------------------------------------- #
# Hosted-LLM selection for the AI-Scientist agents (planner / search / judge).
# Local models hallucinate on structured tool output, so these default to a
# hosted model: gpt-5.6-luna (chosen over gpt-5.2 — comparable results, faster,
# ~2-3x cheaper). Override with DLENS_LLM (e.g. DLENS_LLM=gpt-5.2).
# --------------------------------------------------------------------------- #

DEFAULT_LLM = "gpt-5.6-luna"


def build_llm(model_id: str | None = None) -> Model:
    """Build the hosted LLM for AI-Scientist reasoning (reads OPENAI_API_KEY).

    gpt-5.6* models REQUIRE the OpenAI Responses API: they reject function tools
    on chat completions when reasoning is enabled (HTTP 400), which breaks
    pydantic-ai structured output. Other ids use chat completions.
    """
    mid = (model_id or os.getenv("DLENS_LLM") or DEFAULT_LLM).strip()
    if mid.startswith("gpt-5.6"):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(mid)
    return OpenAIChatModel(mid)
