"""Tests for node server cache control RPC wiring."""

import torch

from distributed_inference.common.serialization import serialize_tensor
from distributed_inference.node.server import NodeServiceImpl
from distributed_inference.proto import inference_pb2


def test_run_forward_passes_cache_flags_to_executor():
    servicer = NodeServiceImpl(node_id="node-1", device_type="cpu")
    captured = {}

    def fake_forward(**kwargs):
        captured.update(kwargs)
        return torch.randn(1, 1, 4)

    servicer.executor.forward = fake_forward
    hidden = torch.tensor([[1]], dtype=torch.long)

    req = inference_pb2.ActivationData(
        hidden_states=inference_pb2.TensorData(
            data=serialize_tensor(hidden),
            shape=list(hidden.shape),
            dtype="int64",
        ),
        request_id="req-1",
        use_cache=True,
        reset_cache=True,
        cache_position=5,
        is_prefill=False,
    )

    _ = servicer.RunForward(req, context=None)
    assert captured["request_id"] == "req-1"
    assert captured["use_cache"] is True
    assert captured["reset_cache"] is True
    assert captured["cache_position"] == 5
    assert captured["is_prefill"] is False


def test_clear_request_cache_rpc_dispatch():
    servicer = NodeServiceImpl(node_id="node-2", device_type="cpu")
    calls = {"clear_one": [], "clear_all": 0}

    servicer.executor.clear_request_cache = lambda request_id: calls["clear_one"].append(request_id)
    servicer.executor.clear_all_cache = lambda: calls.__setitem__("clear_all", calls["clear_all"] + 1)

    _ = servicer.ClearRequestCache(
        inference_pb2.CacheControl(request_id="abc", clear_all=False),
        context=None,
    )
    _ = servicer.ClearRequestCache(
        inference_pb2.CacheControl(clear_all=True),
        context=None,
    )

    assert calls["clear_one"] == ["abc"]
    assert calls["clear_all"] == 1


def test_cancel_request_rpc_marks_request_and_blocks_forward():
    class DummyContext:
        def __init__(self):
            self.code = None
            self.details = ""

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    servicer = NodeServiceImpl(node_id="node-3", device_type="cpu")
    calls = {"forward": 0}

    def fake_forward(**kwargs):
        del kwargs
        calls["forward"] += 1
        return torch.randn(1, 1, 4)

    servicer.executor.forward = fake_forward
    hidden = torch.tensor([[1]], dtype=torch.long)
    req = inference_pb2.ActivationData(
        hidden_states=inference_pb2.TensorData(
            data=serialize_tensor(hidden),
            shape=list(hidden.shape),
            dtype="int64",
        ),
        request_id="cancel-me",
    )

    _ = servicer.CancelRequest(
        inference_pb2.CacheControl(request_id="cancel-me", clear_all=False),
        context=None,
    )

    ctx = DummyContext()
    response = servicer.RunForward(req, context=ctx)
    assert response.hidden_states.data == b""
    assert calls["forward"] == 0
    assert "cancelled" in ctx.details
