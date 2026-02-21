"""Tests for node-agent heartbeat and coordinator registration flow."""

import grpc
import pytest

from distributed_inference.node.agent import NodeAgent
from distributed_inference.proto import inference_pb2


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _FakeServer:
    def start(self):
        return None

    def stop(self, grace=0):
        del grace
        return None

    def wait_for_termination(self):
        return None


class _FakeServicer:
    def _build_status(self):
        return inference_pb2.NodeStatus(
            node_id="node-test",
            status=inference_pb2.NodeStatus.BUSY,
            vram_used_mb=64,
            vram_total_mb=1024,
            timestamp_ms=123,
            active_requests=3,
            queue_depth=2,
            estimated_free_vram_mb=960,
        )


class _FakeChannel:
    def close(self):
        return None


class _FakeStub:
    def __init__(self):
        self.fail_report = False
        self.register_calls = 0
        self.report_calls = 0
        self.last_status = None

    def RegisterNode(self, request, timeout=10):
        del request, timeout
        self.register_calls += 1
        return inference_pb2.RegistrationAck(success=True, message="ok")

    def ReportHealth(self, status, timeout=5):
        del timeout
        self.report_calls += 1
        self.last_status = status
        if self.fail_report:
            raise _FakeRpcError(grpc.StatusCode.NOT_FOUND, "node not registered")
        return inference_pb2.Empty()


def test_heartbeat_failure_marks_node_unregistered(monkeypatch):
    fake_stub = _FakeStub()

    class _Caps:
        vram_mb = 1024
        compute_tflops = 1.0
        bandwidth_mbps = 1000.0
        device_type = "cpu"
        device_name = "fake"
        sram_mb = 0
        latency_ms = 0.0
        effective_bandwidth_mbps = 1000.0

        def summary(self):
            return "fake"

    monkeypatch.setattr(
        "distributed_inference.node.agent.detect_resources",
        lambda **kwargs: _Caps(),
    )
    monkeypatch.setattr(
        "distributed_inference.node.agent.create_node_server",
        lambda **kwargs: _FakeServer(),
    )
    monkeypatch.setattr(
        "distributed_inference.node.agent.grpc.insecure_channel",
        lambda *args, **kwargs: _FakeChannel(),
    )
    monkeypatch.setattr(
        "distributed_inference.node.agent.inference_pb2_grpc.CoordinatorServiceStub",
        lambda channel: fake_stub,
    )
    monkeypatch.setattr(
        "distributed_inference.node.resources.get_vram_usage_mb",
        lambda: 128,
    )

    agent = NodeAgent(
        port=50061,
        coordinator_address="localhost:50050",
        device="cpu",
        node_id="node-test",
    )

    assert agent._register_with_coordinator() is True
    assert agent._registered_with_coordinator is True

    fake_stub.fail_report = True
    with pytest.raises(RuntimeError, match="NOT_FOUND"):
        agent._send_heartbeat()
    assert agent._registered_with_coordinator is False


def test_heartbeat_reports_runtime_telemetry_from_node_servicer(monkeypatch):
    fake_stub = _FakeStub()

    class _Caps:
        vram_mb = 1024
        compute_tflops = 1.0
        bandwidth_mbps = 1000.0
        device_type = "cpu"
        device_name = "fake"
        sram_mb = 0
        latency_ms = 0.0
        effective_bandwidth_mbps = 1000.0

        def summary(self):
            return "fake"

    def _fake_server_factory(**kwargs):
        del kwargs
        server = _FakeServer()
        setattr(server, "_di_servicer", _FakeServicer())
        return server

    monkeypatch.setattr(
        "distributed_inference.node.agent.detect_resources",
        lambda **kwargs: _Caps(),
    )
    monkeypatch.setattr(
        "distributed_inference.node.agent.create_node_server",
        _fake_server_factory,
    )
    monkeypatch.setattr(
        "distributed_inference.node.agent.grpc.insecure_channel",
        lambda *args, **kwargs: _FakeChannel(),
    )
    monkeypatch.setattr(
        "distributed_inference.node.agent.inference_pb2_grpc.CoordinatorServiceStub",
        lambda channel: fake_stub,
    )

    agent = NodeAgent(
        port=50061,
        coordinator_address="localhost:50050",
        device="cpu",
        node_id="node-test",
    )
    assert agent._register_with_coordinator() is True

    agent._send_heartbeat()

    assert fake_stub.last_status is not None
    assert fake_stub.last_status.active_requests == 3
    assert fake_stub.last_status.queue_depth == 2
    assert fake_stub.last_status.estimated_free_vram_mb == 960
