# Distributed Inference System

A prototype distributed inference system where multiple nodes collaboratively run forward-pass inference on a sharded LLM (TinyLlama 1.1B). Designed to validate dynamic model partitioning, measure communication overhead, and test fault tolerance.

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
| **Partitioner**     | VRAM-proportional greedy bin-packing of transformer layers                         |
| **Router**          | Orchestrates sequential forward pass across pipeline stages                        |
| **Fault Tolerance** | Heartbeat health monitoring, activation checkpointing, recovery                    |
| **Profiler**        | Latency, throughput, and communication overhead measurement                        |

## Quick Start

### Install

```bash
cd DistributedInference
pip install -e ".[dev]"
```

### Run the Demo (All-in-One)

```bash
python scripts/run_demo.py --num-nodes 3 --prompt "The future of AI is" --max-tokens 30
```

This spawns a coordinator + 3 simulated nodes (with different VRAM caps), partitions TinyLlama across them, runs inference, and prints metrics.

### Manual Setup

**Terminal 1 — Coordinator:**

```bash
python -m distributed_inference.cli.start_coordinator --port 50050
```

**Terminal 2 — Node 1 (512MB VRAM cap):**

```bash
python -m distributed_inference.cli.start_node --port 50051 --coordinator localhost:50050 --max-vram-mb 512 --node-id node-1
```

**Terminal 3 — Node 2 (1GB VRAM cap):**

```bash
python -m distributed_inference.cli.start_node --port 50052 --coordinator localhost:50050 --max-vram-mb 1024 --node-id node-2
```

**Terminal 4 — Node 3 (1.5GB VRAM cap):**

```bash
python -m distributed_inference.cli.start_node --port 50053 --coordinator localhost:50050 --max-vram-mb 1536 --node-id node-3
```

**Terminal 5 — Run Inference:**

```bash
python -m distributed_inference.cli.run_inference --coordinator localhost:50050 --prompt "Once upon a time" --max-tokens 50
```

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

inference:
  max_tokens: 50
  temperature: 0.7
```

## Key Design Decisions

- **gRPC + Protobuf** for inter-node communication (portable, inspectable)
- **Layer-wise pipeline parallelism** (minimal cross-node dependencies)
- **VRAM-proportional partitioning** (nodes get layers proportional to their VRAM)
- **Simulated VRAM caps** via `--max-vram-mb` for local testing on a single GPU
