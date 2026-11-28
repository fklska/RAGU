import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, TypeAlias

import httpx

DecisionRequest: TypeAlias = dict[str, Any]
DecisionResponse: TypeAlias = dict[str, Any]

__all__ = [
    "DecisionClient",
    "DecisionRequest",
    "DecisionResponse",
    "HTTPDecisionClient",
    "LocalDecisionClient",
]


class DecisionClient(ABC):
    @abstractmethod
    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Execute a structured decision request with state and named questions."""


class HTTPDecisionClient(DecisionClient):
    def __init__(
        self,
        url: str,
        model_name: str | None = None,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        """
        Connect to an HTTP decision API.

        Without ``model_name``, the request is sent as ``state/questions`` JSON.
        With ``model_name``, the FRIDA vLLM envelope ``model/data`` is used.
        A top-level ``data`` response envelope is unwrapped automatically.

        :param url: Full URL of the decision endpoint.
        :param model_name: Optional model identifier for FRIDA vLLM.
        :param api_key: Optional bearer token.
        :param headers: Additional HTTP headers.
        :param timeout: Request timeout in seconds.
        :param client: Optional externally managed HTTP client.
        """
        if not url.strip():
            raise ValueError("url must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.headers = dict(headers or {})
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        payload = (
            {"model": self.model_name, "data": request}
            if self.model_name is not None
            else request
        )
        response = await self._client.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("Decision API must return a JSON object")
        if "answers" not in data and isinstance(data.get("data"), dict):
            data = data["data"]
        return data

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class LocalDecisionClient(DecisionClient):
    def __init__(self, judge: Any):
        """
        Adapt a local decision model.

        The model can be a callable, such as FRIDA ``Judge``, or expose a
        synchronous/asynchronous ``judge(request)`` method, such as ``AsyncJudge``.
        Synchronous inference runs in a worker thread.

        :param judge: Local decision model instance.
        """
        method = getattr(judge, "judge", None)
        if method is None and callable(judge):
            method = judge
        if not callable(method):
            raise TypeError("judge must be callable or expose judge(request)")
        self._method = method

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        if inspect.iscoroutinefunction(self._method):
            result = await self._method(request)
        else:
            result = await asyncio.to_thread(self._method, request)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, dict):
            raise TypeError("Local decision model must return a dictionary")
        return result
