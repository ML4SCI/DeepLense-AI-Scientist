from __future__ import annotations
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.litellm import LiteLLMProvider
import outlines
from llama_cpp import Llama


# Wrapper for OpenAI model (when you want to use OpenAI API)
class OpenAIModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name=model_name)


# Wrapper for Ollama model
class OllamaModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-oss:20b", port: str = "11434"):
        super().__init__(
            model_name=model_name,
            provider=OllamaProvider(base_url=f"http://localhost:{port}/v1"),
        )


# Wrapper for Llama cpp model
class LlamaCppModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-oss:20b", port: str = "8080"):
        super().__init__(
            model_name=model_name,
            provider=OpenAIProvider(base_url=f"http://localhost:{port}/v1"),
        )


# Wrapper for Outlines Llama cpp model - When you want to strictly force output JSON format
class OutlinesLlamaCppModel:
    def __init__(
        self,
        repo_id: str = "ggml-org/gpt-oss-20b-GGUF",
        filename: str = "gpt-oss-20b-mxfp4.gguf",
        *,
        n_ctx: int = 32768,
        chat_format: str | None = None,
    ):
        self.model = outlines.from_llamacpp(
            Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                chat_format=chat_format,
                verbose=False,
            )
        )


# Wrapper for vLLM model [Work in progress]
class VLLMProvider(OpenAIProvider):
    """OpenAI-compatible provider for a local vLLM server."""

    def __init__(
        self, base_url: str = "http://localhost:8000/v1", api_key: str = "local"
    ):
        super().__init__(base_url=base_url, api_key=api_key)


class VLLMModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-oss:20b", port: str = "8000"):
        super().__init__(
            model_name=model_name,
            provider=VLLMProvider(base_url=f"http://localhost:{port}/v1"),
        )


# Wrapper for LiteLLM model [Work in progress]
class LiteLLMModel(OpenAIChatModel):
    def __init__(self, model_name: str = "local-ollama", port: str = "4000"):
        super().__init__(
            model_name=model_name,
            provider=LiteLLMProvider(
                api_base=f"http://127.0.0.1:{port}/v1",
                api_key="local",
            ),
        )
