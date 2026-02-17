"""Run coordinator + web UI with dynamic node onboarding controls.

This script starts:
1) Coordinator (in-process)
2) Web gateway (subprocess)

Nodes can then be added/removed at runtime from either:
- Web UI: Dynamic Node Onboarding panel
- CLI: python -m distributed_inference.cli.manage_nodes ...
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from distributed_inference.common.config import load_config
from distributed_inference.common.logging import get_logger, setup_logging
from distributed_inference.coordinator.orchestrator import Orchestrator

log = get_logger(__name__)


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_web_health(web_url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    health_url = f"{web_url}/health"
    while time.time() < deadline:
        try:
            payload = _request_json("GET", health_url)
            if payload.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for web health at {health_url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start web stack with runtime node onboarding controls"
    )
    parser.add_argument("--coordinator-host", type=str, default="localhost")
    parser.add_argument("--coordinator-port", type=int, default=50050)
    parser.add_argument("--web-host", type=str, default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--initial-nodes", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, component="web-dynamic-demo")

    coordinator_addr = f"{args.coordinator_host}:{args.coordinator_port}"
    web_url = f"http://{args.web_host}:{args.web_port}"

    config = load_config()
    config.coordinator.host = args.coordinator_host
    config.coordinator.port = args.coordinator_port

    orchestrator = Orchestrator(config=config)
    web_proc: subprocess.Popen | None = None

    try:
        log.info(f"[bold blue]Starting coordinator[/] on {coordinator_addr}")
        orchestrator.start(block=False)
        time.sleep(1.0)

        log.info(f"[bold magenta]Starting web gateway[/] on {web_url}")
        web_cmd = [
            sys.executable,
            "-m",
            "distributed_inference.cli.start_web",
            "--host",
            args.web_host,
            "--port",
            str(args.web_port),
            "--coordinator",
            coordinator_addr,
            "--log-level",
            args.log_level,
        ]
        web_proc = subprocess.Popen(
            web_cmd,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
        )

        _wait_for_web_health(web_url, timeout_sec=args.startup_timeout)
        log.info(f"[bold green]Web ready:[/] {web_url}")

        if args.initial_nodes > 0:
            vram_caps = [512, 1024, 1536, 2048]
            for index in range(args.initial_nodes):
                payload = {
                    "node_id": f"node-{index + 1}",
                    "coordinator": coordinator_addr,
                    "device": args.device,
                    "max_vram_mb": vram_caps[index % len(vram_caps)],
                }
                join_url = f"{web_url}/api/nodes/join"
                response = _request_json("POST", join_url, payload)
                log.info(
                    "Started node via web API: "
                    f"{response.get('node_id')} (pid={response.get('pid')})"
                )
                time.sleep(0.5)

            if not orchestrator.wait_for_nodes(args.initial_nodes, timeout=args.startup_timeout):
                raise RuntimeError("Initial nodes did not register in time")

        print("\nDynamic onboarding commands:")
        print(f"  List nodes: python -m distributed_inference.cli.manage_nodes --web-url {web_url} list")
        print(
            f"  Join node:  python -m distributed_inference.cli.manage_nodes --web-url {web_url} "
            "join --device auto --max-vram-mb 1024"
        )
        print(
            f"  Stop node:  python -m distributed_inference.cli.manage_nodes --web-url {web_url} "
            "stop --node-id <node-id>"
        )
        print("Press Ctrl+C to stop coordinator + web + managed nodes.")

        if args.open_browser:
            try:
                webbrowser.open(web_url)
            except Exception:
                log.warning("Could not open browser automatically")

        while True:
            if web_proc.poll() is not None:
                raise RuntimeError("Web gateway exited unexpectedly")
            time.sleep(1.0)

    except urllib.error.URLError as e:
        log.error(f"Web API error: {e}")
    except KeyboardInterrupt:
        log.info("Interrupted, shutting down...")
    finally:
        if web_proc is not None:
            web_proc.terminate()
            try:
                web_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                web_proc.kill()
        orchestrator.stop()
        log.info("[bold green]Stopped all services[/]")


if __name__ == "__main__":
    main()
