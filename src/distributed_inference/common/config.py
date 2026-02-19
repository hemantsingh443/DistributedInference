"""Configuration management for the distributed inference system.

Loads YAML-based configs and provides typed dataclasses for all settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Model-related configuration."""
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float16"  # "float16" or "float32"
    num_layers: int = 22
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_attention_heads: int = 32
    num_kv_heads: int = 4
    max_sequence_length: int = 2048


@dataclass
class CoordinatorConfig:
    """Coordinator settings."""
    host: str = "localhost"
    port: int = 50050
    heartbeat_interval_sec: float = 5.0
    heartbeat_timeout_sec: float = 15.0
    failure_threshold: int = 3  # Missed heartbeats before marking dead
    max_concurrent_requests: int = 4
    max_concurrent_requests_global: int = 4
    max_queue_size: int = 32
    scheduler_policy: str = "balanced"
    fairness_quantum_tokens: int = 16
    tail_latency_guardrail_ms: float = 2500.0
    per_node_vram_safety_margin: float = 0.9
    min_vram_mb: int = 512
    min_compute_tflops: float = 0.5
    gpu_required: bool = False
    rebalance_cooldown_sec: float = 5.0
    node_load_failure_backoff_sec: float = 30.0
    rebalance_drain_timeout_sec: float = 30.0
    allocation_alpha_latency: float = 0.7
    allocation_beta_throughput: float = 0.3
    default_bandwidth_mbps: float = 1000.0
    default_latency_ms: float = 5.0
    memory_safety_margin: float = 0.9


@dataclass
class NodeConfig:
    """Node agent settings."""
    host: str = "localhost"
    port: int = 50051  # Will be overridden per-node
    max_vram_mb: Optional[int] = None  # None = use all available
    device: str = "auto"  # "auto", "cuda", "cpu"
    coordinator_address: str = "localhost:50050"
    max_cached_requests: int = 8
    max_concurrent_lanes: int = 4
    warm_context_pool_size: int = 2
    max_prefill_batch_size: int = 1
    max_cache_tokens_per_request: int = 4096
    cache_eviction_policy: str = "lru"


@dataclass
class InferenceConfig:
    """Inference generation settings."""
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    enable_kv_cache: bool = True


@dataclass
class CommunicationConfig:
    """Communication and transport settings."""
    compress_tensors: bool = False
    max_message_size_mb: int = 256  # gRPC max message size
    grpc_options: dict = field(default_factory=lambda: {
        "grpc.max_send_message_length": 256 * 1024 * 1024,
        "grpc.max_receive_message_length": 256 * 1024 * 1024,
    })


@dataclass
class SystemConfig:
    """Top-level configuration combining all subsystems."""
    model: ModelConfig = field(default_factory=ModelConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from a YAML file.

    If no path is provided, looks for configs/default.yaml relative to
    the project root, then falls back to defaults.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Populated SystemConfig instance.
    """
    config = SystemConfig()

    if config_path is None:
        # Try to find default config
        project_root = Path(__file__).parent.parent.parent.parent
        default_path = project_root / "configs" / "default.yaml"
        if default_path.exists():
            config_path = str(default_path)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # Merge YAML values into dataclass fields
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)

        if "coordinator" in raw:
            for k, v in raw["coordinator"].items():
                if hasattr(config.coordinator, k):
                    setattr(config.coordinator, k, v)

        if "node" in raw:
            for k, v in raw["node"].items():
                if hasattr(config.node, k):
                    setattr(config.node, k, v)

        if "inference" in raw:
            for k, v in raw["inference"].items():
                if hasattr(config.inference, k):
                    setattr(config.inference, k, v)

        if "communication" in raw:
            for k, v in raw["communication"].items():
                if hasattr(config.communication, k):
                    setattr(config.communication, k, v)

    return config
