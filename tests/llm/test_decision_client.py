import json

import httpx
import pytest

from ragu.models.decision_client import (
    HTTPDecisionClient,
    LocalDecisionClient,
)


def make_request():
    return {
        "state": {"text": "Alice and A. Smith"},
        "questions": {
            "same_entity": {
                "type": "noul",
                "instructions": "Do both names denote the same person?",
            }
        },
    }


def make_response():
    return {
        "answers": {
            "same_entity": {
                "type": "noul",
                "noul": 0.8,
            }
        }
    }


async def test_http_client_sends_direct_request_and_auth():
    request_data = make_request()
    response_data = make_response()
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json=response_data)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = HTTPDecisionClient(
            url="http://decisions.test/v1/decisions",
            api_key="secret",
            client=http_client,
        )
        result = await client.decide(request_data)
        await client.close()
        assert not http_client.is_closed

    assert result == response_data
    assert captured["payload"] == request_data
    assert captured["authorization"] == "Bearer secret"


async def test_http_client_supports_frida_vllm_envelope():
    request_data = make_request()
    response_data = make_response()
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"data": response_data})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = HTTPDecisionClient(
            url="http://localhost:8000/pooling",
            model_name="ai-forever/FRIDA-Decisions",
            client=http_client,
        )
        result = await client.decide(request_data)

    assert result == response_data
    assert captured["payload"] == {
        "model": "ai-forever/FRIDA-Decisions",
        "data": request_data,
    }


async def test_local_client_supports_async_judge():
    response_data = make_response()

    class AsyncJudge:
        async def judge(self, request):
            assert request == make_request()
            return response_data

    client = LocalDecisionClient(AsyncJudge())

    assert await client.decide(make_request()) == response_data


async def test_local_client_supports_sync_callable_judge():
    response_data = make_response()

    class SyncJudge:
        def __call__(self, request):
            assert request == make_request()
            return response_data

    client = LocalDecisionClient(SyncJudge())

    assert await client.decide(make_request()) == response_data


async def test_local_client_rejects_invalid_judge_and_response():
    with pytest.raises(TypeError, match="judge must be callable"):
        LocalDecisionClient(object())

    client = LocalDecisionClient(lambda request: ["invalid"])
    with pytest.raises(TypeError, match="must return a dictionary"):
        await client.decide(make_request())


async def test_http_client_validates_json_response():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=["invalid"])
        ),
    ) as http_client:
        client = HTTPDecisionClient(
            url="http://decisions.test",
            client=http_client,
        )
        with pytest.raises(TypeError, match="must return a JSON object"):
            await client.decide(make_request())
