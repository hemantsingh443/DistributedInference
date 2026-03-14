"""CLI for dynamic node and model management via the web gateway API."""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request

from distributed_inference.common.logging import setup_logging


def _request_json(
    method: str,
    web_url: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    base = web_url.rstrip("/")
    url = f"{base}{path}"
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url=url,
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            detail = payload.get("detail", raw)
        except json.JSONDecodeError:
            detail = raw
        raise RuntimeError(f"{e.code} {e.reason}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach web gateway: {e}") from e


def _join(args: argparse.Namespace) -> None:
    payload = {
        "node_id": args.node_id,
        "coordinator": args.coordinator,
        "port": args.port,
        "device": args.device,
        "max_vram_mb": args.max_vram_mb,
        "bandwidth_mbps": args.bandwidth_mbps,
        "latency_ms": args.latency_ms,
        "log_level": args.log_level,
    }
    # Remove unset fields for cleaner server-side defaults.
    payload = {k: v for k, v in payload.items() if v is not None}
    result = _request_json("POST", args.web_url, "/api/nodes/join", payload)
    print(json.dumps(result, indent=2))


def _list(args: argparse.Namespace) -> None:
    result = _request_json("GET", args.web_url, "/api/nodes")
    print(json.dumps(result, indent=2))


def _stop(args: argparse.Namespace) -> None:
    path = "/api/nodes/" + urllib.parse.quote(args.node_id, safe="") + "/stop"
    result = _request_json("POST", args.web_url, path)
    print(json.dumps(result, indent=2))


def _model_list(args: argparse.Namespace) -> None:
    models = _request_json("GET", args.web_url, "/api/models")
    print("Available models:")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")


def _model_switch(args: argparse.Namespace) -> None:
    result = _request_json(
        "POST", args.web_url, "/api/model/switch",
        payload={"model_name": args.name},
    )
    if result.get("success"):
        print(f"Model switch to '{args.name}' initiated.")
        print(f"Status: {result.get('status', '')}")
    else:
        print(f"Model switch failed: {result.get('status', 'unknown error')}")


def _cache(args: argparse.Namespace) -> None:
    """Trigger model weight caching on the coordinator."""
    url = f"{args.web_url.rstrip('/')}/api/model/cache"
    payload = {"model_name": args.name}
    print(f"Requesting coordinator to cache model '{args.name}'...")
    print("This may take several minutes depending on model size and network speed.")
    try:
        data = _request_json("POST", args.web_url, "/api/model/cache", payload)
        if data.get("success"):
            print(f"Success! Model cached at: {data.get('cache_path')}")
        else:
            print(f"Cache request failed: {data.get('status')}")
    except Exception as e:
        print(f"Error caching model: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage dynamic node onboarding via web gateway"
    )
    parser.add_argument(
        "--web-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Web gateway base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    join_parser = subparsers.add_parser("join", help="Launch and join a new node")
    join_parser.add_argument("--node-id", type=str, default=None)
    join_parser.add_argument("--coordinator", type=str, default=None)
    join_parser.add_argument("--port", type=int, default=None)
    join_parser.add_argument("--device", type=str, default="auto")
    join_parser.add_argument("--max-vram-mb", type=int, default=None)
    join_parser.add_argument("--bandwidth-mbps", type=float, default=None)
    join_parser.add_argument("--latency-ms", type=float, default=None)
    join_parser.set_defaults(func=_join)

    list_parser = subparsers.add_parser("list", help="List managed nodes")
    list_parser.set_defaults(func=_list)

    stop_parser = subparsers.add_parser("stop", help="Stop a managed node")
    stop_parser.add_argument("--node-id", type=str, required=True)
    stop_parser.set_defaults(func=_stop)

    # --- Model management subcommands ---
    model_list_parser = subparsers.add_parser("model-list", help="List available models")
    model_list_parser.set_defaults(func=_model_list)

    model_switch_parser = subparsers.add_parser("model-switch", help="Switch the active model")
    model_switch_parser.add_argument(
        "--name", type=str, required=True,
        help="Model name to switch to (e.g. Qwen/Qwen2-0.5B-Instruct)"
    )
    model_switch_parser.set_defaults(func=_model_switch)

    model_cache_parser = subparsers.add_parser("model-cache", help="Pre-cache a model's weights on the coordinator")
    model_cache_parser.add_argument(
        "--name", type=str, required=True,
        help="Model name to cache (e.g. Qwen/Qwen2-0.5B-Instruct)"
    )
    model_cache_parser.set_defaults(func=_cache)

    args = parser.parse_args()
    setup_logging(level=args.log_level, component="manage-cli")
    args.func(args)


if __name__ == "__main__":
    main()
