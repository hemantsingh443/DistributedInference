"""CLI entry point for starting the coordinator."""

import argparse
import sys

from distributed_inference.common.config import load_config
from distributed_inference.common.logging import setup_logging
from distributed_inference.coordinator.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Start the Distributed Inference Coordinator"
    )
    parser.add_argument(
        "--port", type=int, default=50050,
        help="Port for the coordinator gRPC server (default: 50050)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--min-nodes", type=int, default=1,
        help="Minimum nodes to wait for before model setup (default: 1)"
    )
    parser.add_argument(
        "--auto-setup", action="store_true",
        help="Automatically set up model once min-nodes register"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the initial active model name (e.g. Qwen/Qwen2-0.5B-Instruct)"
    )
    parser.add_argument(
        "--enable-concurrent-scheduler",
        action="store_true",
        help="Enable multi-user concurrent scheduler and fairness runtime",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level, component="coordinator")

    config = load_config(args.config)
    config.coordinator.port = args.port
    config.coordinator.host = args.host
    if args.model:
        config.coordinator.active_model_name = args.model
    if args.enable_concurrent_scheduler:
        config.coordinator.enable_concurrent_scheduler = True

    import asyncio

    async def run_coordinator():
        orchestrator = Orchestrator(config=config)
        if args.auto_setup:
            await orchestrator.start()
            if await orchestrator.wait_for_nodes(args.min_nodes, timeout=120):
                await orchestrator.setup_model()
            
            try:
                await orchestrator._server.wait_for_termination()
            except KeyboardInterrupt:
                await orchestrator.stop()
        else:
            await orchestrator.start()
            try:
                await orchestrator._server.wait_for_termination()
            except KeyboardInterrupt:
                await orchestrator.stop()

    asyncio.run(run_coordinator())


if __name__ == "__main__":
    main()
