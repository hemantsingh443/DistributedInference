"""CLI entry point for starting the web inference gateway."""

import argparse

import uvicorn

from distributed_inference.common.logging import setup_logging
from distributed_inference.web import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Start the Distributed Inference Web Gateway"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind the web server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the web server (default: 8000)"
    )
    parser.add_argument(
        "--coordinator", type=str, default="localhost:50050",
        help="Coordinator address (default: localhost:50050)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, component="web")

    app = create_app(default_coordinator=args.coordinator)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
