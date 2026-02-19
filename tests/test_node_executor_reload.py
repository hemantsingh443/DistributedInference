"""Tests for shard reload behavior in ShardExecutor."""

import torch
import torch.nn as nn

from distributed_inference.node.executor import ShardExecutor


class _DummyInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.norm = nn.Identity()


class _DummyFullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyInnerModel()
        self.lm_head = nn.Identity()
        self.config = type("Cfg", (), {})()


def _dummy_config(*_a, **_kw):
    """Return a minimal config-like object for test mocking."""
    return type("Cfg", (), {"num_hidden_layers": 4, "use_cache": False})()


def _dummy_from_config(config, *_a, **_kw):
    """Return a dummy full model shell for test mocking."""
    return _DummyFullModel()


def _dummy_resolve_safetensor_files(_self, model_name, needed_prefixes):
    """Mock _resolve_safetensor_files to return empty mapping."""
    return {}


def test_load_shard_unloads_previous_model_before_reload(monkeypatch):
    executor = ShardExecutor(device_type="cpu")
    executor._loaded = True
    executor.model = nn.Identity()

    unload_calls = {"count": 0}
    original_unload = executor.unload

    def _counting_unload():
        unload_calls["count"] += 1
        original_unload()

    monkeypatch.setattr(executor, "unload", _counting_unload)
    monkeypatch.setattr(
        "distributed_inference.node.executor.AutoConfig.from_pretrained",
        _dummy_config,
    )
    monkeypatch.setattr(
        "distributed_inference.node.executor.AutoModelForCausalLM.from_config",
        _dummy_from_config,
    )
    monkeypatch.setattr(
        ShardExecutor,
        "_resolve_safetensor_files",
        _dummy_resolve_safetensor_files,
    )
    monkeypatch.setattr(
        "distributed_inference.node.executor.get_vram_usage_mb",
        lambda: 0,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    _ = executor.load_shard(
        model_name="dummy",
        start_layer=0,
        end_layer=2,
        has_embedding=True,
        has_lm_head=False,
        dtype="float32",
    )

    assert unload_calls["count"] == 1
    assert executor.loaded is True


def test_selective_weight_key_prefixes():
    """Verify _get_shard_weight_keys returns correct prefixes."""
    keys = ShardExecutor._get_shard_weight_keys(
        start_layer=3, end_layer=6,
        has_embedding=True, has_lm_head=False,
    )
    assert "model.layers.3." in keys
    assert "model.layers.4." in keys
    assert "model.layers.5." in keys
    assert "model.layers.6." not in keys
    assert "model.embed_tokens." in keys
    assert "lm_head." not in keys
    assert "model.norm." not in keys


def test_selective_weight_key_prefixes_lm_head():
    """Verify lm_head prefixes are included when has_lm_head=True."""
    keys = ShardExecutor._get_shard_weight_keys(
        start_layer=10, end_layer=12,
        has_embedding=False, has_lm_head=True,
    )
    assert "model.layers.10." in keys
    assert "model.layers.11." in keys
    assert "model.embed_tokens." not in keys
    assert "lm_head." in keys
    assert "model.norm." in keys


def test_remap_layer_key():
    """Verify layer index remapping in weight keys."""
    assert ShardExecutor._remap_layer_key(
        "model.layers.15.self_attn.q_proj.weight", start_layer=15
    ) == "model.layers.0.self_attn.q_proj.weight"

    assert ShardExecutor._remap_layer_key(
        "model.layers.3.mlp.gate_proj.weight", start_layer=2
    ) == "model.layers.1.mlp.gate_proj.weight"

    # Non-layer keys should pass through unchanged
    assert ShardExecutor._remap_layer_key(
        "model.embed_tokens.weight", start_layer=5
    ) == "model.embed_tokens.weight"

    assert ShardExecutor._remap_layer_key(
        "lm_head.weight", start_layer=5
    ) == "lm_head.weight"
