"""All-in-one web demo script for distributed inference.

Starts:
1) Coordinator (in-process)
2) N node subprocesses
3) Web gateway subprocess

Then keeps the stack running until interrupted.

Usage:
    python scripts/run_web_demo.py
    python scripts/run_web_demo.py --num-nodes 3 --web-port 8000
    python scripts/run_web_demo.py --open-browser
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from distributed_inference.common.config import load_config
from distributed_inference.common.logging import get_logger, setup_logging
from distributed_inference.coordinator.orchestrator import Orchestrator

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run coordinator + nodes + web UI in one command"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=3,
        help="Number of simulated nodes (default: 3)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for node inference (default: auto)"
    )
    parser.add_argument(
        "--coordinator-port", type=int, default=50050,
        help="Coordinator port (default: 50050)"
    )
    parser.add_argument(
        "--web-host", type=str, default="127.0.0.1",
        help="Web host bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--web-port", type=int, default=8000,
        help="Web UI port (default: 8000)"
    )
    parser.add_argument(
        "--startup-timeout", type=float, default=120.0,
        help="Seconds to wait for node registration (default: 120)"
    )
    parser.add_argument(
        "--open-browser", action="store_true",
        help="Open the web UI in your default browser"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, component="web-demo")

    vram_caps = {
        1: [2048],
        2: [1024, 1024],
        3: [512, 1024, 1536],
        4: [512, 512, 1024, 1024],
    }
    node_vrams = vram_caps.get(args.num_nodes, [1024] * args.num_nodes)

    coordinator_addr = f"localhost:{args.coordinator_port}"
    web_url = f"http://{args.web_host}:{args.web_port}"
    processes = []
    orchestrator = None

    try:
        log.info("[bold blue]Starting coordinator in-process...[/]")
        config = load_config()
        config.coordinator.port = args.coordinator_port

        orchestrator = Orchestrator(config=config)
        orchestrator.start(block=False)
        time.sleep(1)

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
                "--log-level", args.log_level,
            ]
            node_proc = subprocess.Popen(
                node_cmd,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
            )
            processes.append(node_proc)
            time.sleep(1)

        log.info(f"Waiting for {args.num_nodes} nodes to register...")
        if not orchestrator.wait_for_nodes(args.num_nodes, timeout=args.startup_timeout):
            raise RuntimeError(
                f"Timed out waiting for {args.num_nodes} nodes to register"
            )

        log.info("[bold yellow]Partitioning model and loading shards...[/]")
        orchestrator.setup_model()

        log.info("[bold magenta]Starting web gateway...[/]")
        web_cmd = [
            sys.executable, "-m", "distributed_inference.cli.start_web",
            "--host", args.web_host,
            "--port", str(args.web_port),
            "--coordinator", coordinator_addr,
            "--log-level", args.log_level,
        ]
        web_proc = subprocess.Popen(
            web_cmd,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
        )
        processes.append(web_proc)

        time.sleep(1)
        log.info(f"[bold green]Web UI ready:[/] {web_url}")
        log.info("Press Ctrl+C to stop all services")

        if args.open_browser:
            try:
                webbrowser.open(web_url)
            except Exception:
                log.warning("Could not open browser automatically")

        while True:
            # Keep process alive and fail fast if web process crashes.
            if web_proc.poll() is not None:
                raise RuntimeError("Web gateway exited unexpectedly")
            time.sleep(1.0)

    except KeyboardInterrupt:
        log.info("\nInterrupted, shutting down...")
    except Exception as e:
        log.error(f"Demo failed: {e}")
    finally:
        if orchestrator:
            try:
                orchestrator.stop()
            except Exception:
                pass

        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()

        log.info("[bold green]All services stopped[/]")


if __name__ == "__main__":
    main()
