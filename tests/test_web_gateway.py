"""Tests for the FastAPI web gateway SSE layer."""

import io
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


class _FakeProcess:
    _pid_counter = 9999

    def __init__(self, *, logs: str = ""):
        type(self)._pid_counter += 1
        self.pid = type(self)._pid_counter
        self._exit_code = None
        self.stdout = io.StringIO(logs)

    def poll(self):
        return self._exit_code

    def terminate(self):
        self._exit_code = 0

    def wait(self, timeout=None):
        del timeout
        if self._exit_code is None:
            self._exit_code = 0
        return self._exit_code

    def kill(self):
        self._exit_code = -9


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


def test_dynamic_node_api_join_list_stop(monkeypatch):
    monkeypatch.setattr(web_app_module, "_is_port_available", lambda port: True)
    monkeypatch.setattr(
        web_app_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(
            logs="Registered successfully: Node web-node-1 registered successfully\n"
        ),
    )

    app = web_app_module.create_app(default_coordinator="localhost:50050")
    client = TestClient(app)

    join_payload = {
        "node_id": "web-node-1",
        "port": 50061,
        "device": "cpu",
        "max_vram_mb": 1024,
        "bandwidth_mbps": 900,
        "latency_ms": 8,
    }
    join_resp = client.post("/api/nodes/join", json=join_payload)
    assert join_resp.status_code == 200
    joined = join_resp.json()
    assert joined["success"] is True
    assert joined["node_id"] == "web-node-1"
    assert joined["port"] == 50061
    assert joined["admitted"] is True

    list_resp = client.get("/api/nodes")
    assert list_resp.status_code == 200
    listed = list_resp.json()["nodes"]
    assert len(listed) == 1
    assert listed[0]["node_id"] == "web-node-1"
    assert listed[0]["running"] is True

    stop_resp = client.post("/api/nodes/web-node-1/stop")
    assert stop_resp.status_code == 200
    stopped = stop_resp.json()
    assert stopped["success"] is True
    assert stopped["node_id"] == "web-node-1"

    list_after = client.get("/api/nodes").json()["nodes"]
    assert list_after == []


def test_dynamic_node_join_rejection_surfaces_error(monkeypatch):
    monkeypatch.setattr(web_app_module, "_is_port_available", lambda port: True)
    monkeypatch.setattr(
        web_app_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(
            logs=(
                "Registration failed: Node web-node-2 rejected "
                "(Insufficient VRAM: 256MB < minimum 512MB)\n"
            )
        ),
    )

    app = web_app_module.create_app(default_coordinator="localhost:50050")
    client = TestClient(app)
    join_resp = client.post(
        "/api/nodes/join",
        json={
            "node_id": "web-node-2",
            "port": 50062,
            "device": "cpu",
            "max_vram_mb": 256,
        },
    )

    assert join_resp.status_code == 400
    assert "Insufficient VRAM" in join_resp.json()["detail"]
    listed = client.get("/api/nodes").json()["nodes"]
    assert listed == []
