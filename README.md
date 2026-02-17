# Distributed Inference System

A prototype distributed inference system where multiple nodes collaboratively run forward-pass inference on a sharded LLM (TinyLlama 1.1B). Designed to validate dynamic model partitioning, measure communication overhead, and test fault tolerance.

> **Note:** request-scoped KV cache is supported for TinyLlama/Llama-like models on distributed layer shards. `executor.py` is currently optimized for TinyLlama 1.1B or similar architecture.

## Architecture

```
Client → Coordinator → [Node 1] → [Node 2] → [Node 3] → Output
              │            ↑           ↑           ↑
              └─ Partitioner + Scheduler + Health Monitor
```

**Pipeline parallelism**: The model is split into contiguous layer ranges. Each node runs its assigned layers and passes activations to the next node via gRPC.

### Components

| Component           | Description                                                                        |
| ------------------- | ---------------------------------------------------------------------------------- |
| **Node Agent**      | Discovers GPU/CPU resources, loads model shards, serves gRPC forward-pass requests |
| **Coordinator**     | Registers nodes, partitions model, routes activations, manages health              |
| **Partitioner**     | Compute-aware, memory-safe layer allocation using VRAM + compute + network signals |
| **Router**          | Orchestrates sequential forward pass across pipeline stages                        |
| **Fault Tolerance** | Heartbeat health monitoring, activation checkpointing, recovery                    |
| **Profiler**        | Latency, throughput, and communication overhead measurement                        |

## Quick Start

### Install

```bash
cd DistributedInference
pip install -e ".[dev]"
```

### Run Web + Dynamic Onboarding

```bash
python scripts/run_web_dynamic_demo.py --web-port 8000 --initial-nodes 0 --open-browser
```

This starts coordinator + web UI, then lets you add/remove nodes at runtime from:
- **Web UI**: Dynamic Node Onboarding panel
- **CLI**: `python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 ...`

> **Capacity note:** the model is loaded only when aggregate feasible VRAM is enough. If nodes are admitted but total capacity is still insufficient, inference remains unavailable until more capacity joins.

CLI examples:

```bash
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 list
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 stop --node-id web-node-1234abcd
```

### CLI Dynamic Workflow

**Terminal 1 — Coordinator:**

```bash
python -m distributed_inference.cli.start_coordinator --port 50050
```

**Terminal 2 — Web gateway (required for `manage_nodes`):**

```bash
python -m distributed_inference.cli.start_web --host 127.0.0.1 --port 8000 --coordinator localhost:50050
```

**Terminal 3 — Dynamically add nodes from CLI:**

```bash
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
```

**List / stop dynamically managed nodes:**

```bash
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 list
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 stop --node-id <node-id>
```

**Run inference from CLI:**

```bash
python -m distributed_inference.cli.run_inference --coordinator localhost:50050 --prompt "Once upon a time" --max-tokens 50
```

### Web Inference Console (Live Hop + Token Stream)

Open `http://127.0.0.1:8000` in your browser to:
- submit prompts from a UI
- watch generated text stream token-by-token
- view a per-hop timeline (node, layer range, latency) for every decode step

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/distributed_inference/
├── common/         # Config, serialization, logging
├── node/           # Node agent, executor, gRPC server, resource detection
├── coordinator/    # Orchestrator, registry, partitioner, scheduler, router
├── fault_tolerance/# Health monitor, checkpointing, recovery
├── pipeline/       # High-level inference pipeline API
├── benchmarks/     # Performance profiler
├── cli/            # CLI entry points
├── web/            # FastAPI + HTMX web UI and SSE gateway
└── proto/          # Generated gRPC/Protobuf stubs
```

## Configuration

Edit `configs/default.yaml` or pass `--config path/to/config.yaml`:

```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype: "float16"
  num_layers: 22

coordinator:
  port: 50050
  heartbeat_interval_sec: 5.0
  min_vram_mb: 512
  min_compute_tflops: 0.5
  gpu_required: false
  rebalance_cooldown_sec: 5.0
  memory_safety_margin: 0.9

inference:
  max_tokens: 50
  temperature: 0.7
  enable_kv_cache: true

node:
  max_cached_requests: 1
  max_cache_tokens_per_request: 4096
  cache_eviction_policy: "lru"
```

### KV Cache Performance Expectations

With distributed KV cache enabled, prefill still processes the full prompt once, but decode steps reuse per-node cache instead of recomputing the full prefix every token.

- **First token latency:** usually similar to no-cache mode.
- **Decode token latency:** typically much lower than no-cache mode.
- **Throughput:** commonly improves by about **2x to 5x** in practical runs.
- **End-to-end latency (longer generations):** often reduced by around **30% to 70%**.

Actual gains depend on prompt length, generated tokens, network overhead, and GPU/CPU characteristics.

### Quick A/B Measurement

Run the same prompt twice and compare reported `tok/s` and total latency:

1. Start coordinator + web dynamic stack:
```bash
python scripts/run_web_dynamic_demo.py --web-port 8000 --initial-nodes 0
```

2. Add sufficient capacity (example: three 1GB-cap nodes):
```bash
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
python -m distributed_inference.cli.manage_nodes --web-url http://127.0.0.1:8000 join --device auto --max-vram-mb 1024
```

3. KV cache enabled inference:
```bash
python -m distributed_inference.cli.run_inference --coordinator localhost:50050 --prompt "The future of AI is" --max-tokens 100
```

4. KV cache disabled (temporary config override):
Set `inference.enable_kv_cache: false` in your config and rerun the same command.

## Key Design Decisions

- **gRPC + Protobuf** for inter-node communication (portable, inspectable)
- **Streaming inference RPC** for live token/hop telemetry to web clients
- **Layer-wise pipeline parallelism** (minimal cross-node dependencies)
- **Request-scoped distributed KV cache** (prefill + decode with per-node cache reuse)
- **Admission-controlled onboarding** (threshold checks before node participation)
- **Compute-aware partitioning** (VRAM fit + compute + network-aware layer allocation)
- **Simulated VRAM caps** via `--max-vram-mb` for local testing on a single GPU
