"""Tests for the FastAPI web gateway SSE layer."""

import importlib

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from distributed_inference.proto import inference_pb2

web_app_module = importlib.import_module("distributed_inference.web.app")


class _DummyChannel:
    def close(self):
        return None


class _FakeStub:
    def SubmitInferenceStream(self, request, timeout=None):
        del timeout
        yield inference_pb2.InferenceEvent(
            request_id=request.request_id,
            timestamp_ms=1,
            hop=inference_pb2.HopEvent(
                step=0,
                hop_index=0,
                node_id="node-1",
                address="localhost:50051",
                start_layer=0,
                end_layer=8,
                hop_latency_ms=11.5,
            ),
        )
        yield inference_pb2.InferenceEvent(
            request_id=request.request_id,
            timestamp_ms=2,
            token=inference_pb2.TokenEvent(
                step=0,
                token_id=4,
                token_text=" world",
                accumulated_text="Hello world",
            ),
        )
        yield inference_pb2.InferenceEvent(
            request_id=request.request_id,
            timestamp_ms=3,
            completed=inference_pb2.InferenceResponse(
                request_id=request.request_id,
                generated_text="Hello world",
                tokens_generated=1,
                total_latency_ms=33.0,
                tokens_per_second=30.3,
                per_hop_latency_ms=[11.5],
            ),
        )


def test_sse_stream_and_run_log_capture(monkeypatch):
    monkeypatch.setattr(
        web_app_module.grpc,
        "insecure_channel",
        lambda *args, **kwargs: _DummyChannel(),
    )
    monkeypatch.setattr(
        web_app_module.inference_pb2_grpc,
        "CoordinatorServiceStub",
        lambda channel: _FakeStub(),
    )

    app = web_app_module.create_app(default_coordinator="localhost:50050")
    client = TestClient(app)

    with client.stream(
        "GET",
        "/api/stream",
        params={"prompt": "hello", "request_id": "req-web-1"},
    ) as response:
        text = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: start" in text
    assert "event: hop" in text
    assert "event: token" in text
    assert "event: completed" in text

    run_response = client.get("/api/runs/req-web-1")
    assert run_response.status_code == 200
    payload = run_response.json()
    assert payload["request_id"] == "req-web-1"
    assert payload["events"][0]["type"] == "start"
    assert payload["events"][-1]["type"] == "completed"
