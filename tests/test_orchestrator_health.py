"""Tests for coordinator health-report handling edge cases."""

from distributed_inference.coordinator.orchestrator import (
    CoordinatorServiceImpl,
    Orchestrator,
)
from distributed_inference.proto import inference_pb2


class DummyContext:
    """Minimal gRPC-like context for unit testing status/error setting."""

    def __init__(self):
        self.code = None
        self.details = ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_report_health_rejects_unregistered_node():
    orchestrator = Orchestrator()
    service = CoordinatorServiceImpl(orchestrator)
    context = DummyContext()

    _ = service.ReportHealth(
        inference_pb2.NodeStatus(
            node_id="ghost-node",
            status=inference_pb2.NodeStatus.READY,
            vram_used_mb=0,
            vram_total_mb=0,
            timestamp_ms=1,
        ),
        context,
    )

    assert context.code is not None
    assert "not registered" in context.details
