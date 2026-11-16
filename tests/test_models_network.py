import httpx
import pytest

from dlens.agents._models import LiteLLMModel, VLLMModel


@pytest.fixture
def block_http_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected HTTP request during model construction")

    monkeypatch.setattr(httpx.Client, "send", fail_request)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail_request)


@pytest.mark.parametrize(
    "model_class",
    [VLLMModel, LiteLLMModel],
    ids=["vllm", "litellm"],
)
def test_model_construction_does_not_send_http_requests(
    model_class: type,
    block_http_requests: None,
) -> None:
    model_class()
