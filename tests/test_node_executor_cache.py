"""Unit tests for request-scoped KV cache behavior in ShardExecutor."""

import pytest
import torch
import torch.nn as nn

from distributed_inference.node.executor import ShardExecutor


class DummyOutput:
    """Simple output object that mimics transformers model output."""

    def __init__(self, last_hidden_state, past_key_values):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class DummyModel:
    """Model stub that records calls and returns synthetic caches."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        **kwargs,
    ):
        del attention_mask, position_ids, kwargs
        if input_ids is not None:
            batch, seq_len = input_ids.shape
            hidden = torch.zeros((batch, seq_len, 4), dtype=torch.float32)
        else:
            hidden = inputs_embeds.clone()

        call = {
            "past_key_values": past_key_values,
            "cache_position": cache_position.clone() if cache_position is not None else None,
            "use_cache": use_cache,
        }
        self.calls.append(call)

        return DummyOutput(
            last_hidden_state=hidden,
            past_key_values={"call_id": len(self.calls)},
        )


def make_executor(max_cached_requests=1) -> ShardExecutor:
    executor = ShardExecutor(
        device_type="cpu",
        max_cached_requests=max_cached_requests,
        max_cache_tokens_per_request=16,
    )
    executor.model = DummyModel()
    executor.lm_head = None
    executor.has_embedding = False
    executor.has_lm_head = False
    executor._loaded = True
    return executor


def test_prefill_then_decode_reuses_cache():
    executor = make_executor()

    _ = executor.forward(
        hidden_states=torch.randn(1, 3, 4),
        request_id="req-1",
        use_cache=True,
        reset_cache=True,
        is_prefill=True,
        cache_position=2,
    )
    _ = executor.forward(
        hidden_states=torch.randn(1, 1, 4),
        request_id="req-1",
        use_cache=True,
        is_prefill=False,
        cache_position=3,
    )

    assert len(executor.model.calls) == 2
    assert executor.model.calls[0]["past_key_values"] is None
    assert executor.model.calls[1]["past_key_values"] == {"call_id": 1}
    assert executor._kv_cache_by_request["req-1"].tokens_seen == 4
    assert executor.model.calls[1]["cache_position"].tolist() == [3]


def test_decode_without_prefill_raises_cache_miss():
    executor = make_executor()
    with pytest.raises(RuntimeError, match="KV cache miss"):
        executor.forward(
            hidden_states=torch.randn(1, 1, 4),
            request_id="req-miss",
            use_cache=True,
            is_prefill=False,
            cache_position=0,
        )


def test_lru_eviction_when_max_cached_requests_exceeded():
    executor = make_executor(max_cached_requests=1)

    _ = executor.forward(
        hidden_states=torch.randn(1, 2, 4),
        request_id="req-a",
        use_cache=True,
        reset_cache=True,
        is_prefill=True,
        cache_position=1,
    )
    _ = executor.forward(
        hidden_states=torch.randn(1, 2, 4),
        request_id="req-b",
        use_cache=True,
        reset_cache=True,
        is_prefill=True,
        cache_position=1,
    )

    assert "req-a" not in executor._kv_cache_by_request
    assert "req-b" in executor._kv_cache_by_request


def test_renumber_layer_indices_for_sliced_layers():
    class DummyAttn(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

    class DummyLayer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            self.self_attn = DummyAttn(layer_idx)

    layers = nn.ModuleList([DummyLayer(10), DummyLayer(11), DummyLayer(12)])
    ShardExecutor._renumber_layer_indices(layers)

    assert [layer.layer_idx for layer in layers] == [0, 1, 2]
    assert [layer.self_attn.layer_idx for layer in layers] == [0, 1, 2]
