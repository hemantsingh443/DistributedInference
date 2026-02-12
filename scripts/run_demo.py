"""All-in-one demo script for the distributed inference system.

Spawns a coordinator + multiple node processes, partitions TinyLlama
across them, runs inference, and reports metrics.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --num-nodes 3 --prompt "Once upon a time"
    python scripts/run_demo.py --device cpu --max-tokens 30
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from distributed_inference.common.logging import setup_logging, get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Demo")
    parser.add_argument(
        "--num-nodes", type=int, default=3,
        help="Number of simulated nodes (default: 3)"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The future of artificial intelligence is",
        help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=30,
        help="Max tokens to generate (default: 30)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference (default: auto)"
    )
    parser.add_argument(
        "--coordinator-port", type=int, default=50050,
        help="Coordinator port (default: 50050)"
    )

    args = parser.parse_args()

    setup_logging(level="INFO", component="demo")

    # Define VRAM caps for heterogeneous simulation
    # These simulate nodes with different amounts of VRAM
    vram_caps = {
        1: [2048],              # 1 node: all VRAM
        2: [1024, 1024],        # 2 nodes: even split
        3: [512, 1024, 1536],   # 3 nodes: small/medium/large
        4: [512, 512, 1024, 1024],
    }
    node_vrams = vram_caps.get(args.num_nodes, [1024] * args.num_nodes)

    coordinator_addr = f"localhost:{args.coordinator_port}"
    processes = []

    try:
        # ─── Start Coordinator ───
        log.info("[bold blue]Starting coordinator...[/]")
        coord_cmd = [
            sys.executable, "-m", "distributed_inference.cli.start_coordinator",
            "--port", str(args.coordinator_port),
            "--log-level", "INFO",
        ]
        coord_proc = subprocess.Popen(
            coord_cmd,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
        )
        processes.append(coord_proc)
        time.sleep(2)  # Give coordinator time to start

        # ─── Start Nodes ───
        node_base_port = args.coordinator_port + 1
        for i in range(args.num_nodes):
            port = node_base_port + i
            vram = node_vrams[i]
            node_id = f"node-{i+1}"

            log.info(
                f"[bold cyan]Starting {node_id}[/] on port {port} "
                f"(VRAM cap: {vram}MB)"
            )

            node_cmd = [
                sys.executable, "-m", "distributed_inference.cli.start_node",
                "--port", str(port),
                "--coordinator", coordinator_addr,
                "--max-vram-mb", str(vram),
                "--device", args.device,
                "--node-id", node_id,
                "--log-level", "INFO",
            ]
            node_proc = subprocess.Popen(
                node_cmd,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
            )
            processes.append(node_proc)
            time.sleep(1)

        # Wait for all nodes to register
        log.info(f"Waiting for {args.num_nodes} nodes to register...")
        time.sleep(5)

        # ─── Set up model and run inference via Python API ───
        log.info("[bold magenta]Setting up model and running inference...[/]")

        from distributed_inference.common.config import load_config
        from distributed_inference.coordinator.orchestrator import Orchestrator
        from distributed_inference.benchmarks.profiler import InferenceProfiler

        import grpc
        from distributed_inference.proto import inference_pb2
        from distributed_inference.proto import inference_pb2_grpc

        # Connect to coordinator and set up model
        options = [
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(coordinator_addr, options=options)
        stub = inference_pb2_grpc.CoordinatorServiceStub(channel)

        # First, we need the coordinator to set up the model
        # We do this by creating a local orchestrator instance that connects
        # to the same registry
        # For the demo, we'll use the coordinator directly
        config = load_config()
        config.coordinator.port = args.coordinator_port

        # Use the orchestrator directly for setup
        orchestrator = Orchestrator(config=config)
        orchestrator.start(block=False)

        # Wait for nodes to register with this orchestrator
        time.sleep(3)
        orchestrator.wait_for_nodes(args.num_nodes, timeout=30)

        # Set up model
        log.info("[bold yellow]Partitioning model and loading shards...[/]")
        orchestrator.setup_model()

        # Run inference
        profiler = InferenceProfiler()

        log.info(f"\n[bold]Prompt:[/] {args.prompt}")
        log.info("[bold yellow]Running distributed inference...[/]")

        start = time.time()
        result = orchestrator.run_inference(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            request_id=uuid.uuid4().hex[:8],
        )
        total_ms = (time.time() - start) * 1000

        # Record metrics
        profiler.record_run(
            request_id="demo",
            prompt_tokens=len(args.prompt.split()),
            generated_tokens=result.tokens_generated,
            total_latency_ms=result.total_latency_ms,
            per_hop_latencies=list(result.per_hop_latency_ms),
        )

        # Print results
        print("\n" + "=" * 60)
        print("🤖 DISTRIBUTED INFERENCE RESULT")
        print("=" * 60)
        print(f"\n{result.generated_text}\n")
        print("=" * 60)
        print(f"📊 Tokens generated: {result.tokens_generated}")
        print(f"⏱️  Total latency:    {result.total_latency_ms:.1f}ms")
        print(f"🚀 Throughput:       {result.tokens_per_second:.2f} tok/s")
        print(f"🖥️  Nodes used:       {args.num_nodes}")

        if result.per_hop_latency_ms:
            print(f"📡 Hop latencies:    {[f'{l:.0f}ms' for l in result.per_hop_latency_ms[:10]]}")

        profiler.print_comparison()

        # Clean up
        orchestrator.stop()

    except KeyboardInterrupt:
        log.info("\nDemo interrupted")

    except Exception as e:
        log.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up all processes
        log.info("Shutting down all processes...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()

        log.info("[bold green]Demo complete![/]")


if __name__ == "__main__":
    main()
