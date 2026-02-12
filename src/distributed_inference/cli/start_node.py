"""CLI entry point for starting a node agent."""

import argparse

from distributed_inference.common.logging import setup_logging
from distributed_inference.node.agent import NodeAgent


def main():
    parser = argparse.ArgumentParser(
        description="Start a Distributed Inference Node"
    )
    parser.add_argument(
        "--port", type=int, required=True,
        help="Port for this node's gRPC server"
    )
    parser.add_argument(
        "--coordinator", type=str, default="localhost:50050",
        help="Coordinator address (default: localhost:50050)"
    )
    parser.add_argument(
        "--max-vram-mb", type=int, default=None,
        help="Simulated VRAM cap in MB (default: use all available)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--node-id", type=str, default=None,
        help="Custom node ID (default: auto-generated)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level, component=f"node:{args.port}")

    agent = NodeAgent(
        port=args.port,
        coordinator_address=args.coordinator,
        max_vram_mb=args.max_vram_mb,
        device=args.device,
        node_id=args.node_id,
    )

    agent.start(block=True)


if __name__ == "__main__":
    main()
