"""CLI entry point for submitting an inference request."""

import argparse
import uuid

import grpc

from distributed_inference.common.logging import setup_logging, get_logger
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Submit an inference request to the coordinator"
    )
    parser.add_argument(
        "--coordinator", type=str, default="localhost:50050",
        help="Coordinator address (default: localhost:50050)"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Input prompt for text generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling threshold (default: 0.9)"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling parameter (default: 50)"
    )

    args = parser.parse_args()

    setup_logging(component="client")

    options = [
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(args.coordinator, options=options)
    stub = inference_pb2_grpc.CoordinatorServiceStub(channel)

    request = inference_pb2.InferenceRequest(
        request_id=uuid.uuid4().hex[:8],
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    log.info(f"Sending inference request: '{args.prompt[:50]}...'")

    try:
        response = stub.SubmitInference(request, timeout=300)

        print("\n" + "=" * 60)
        print("GENERATED TEXT:")
        print("=" * 60)
        print(response.generated_text)
        print("=" * 60)
        print(f"Tokens generated: {response.tokens_generated}")
        print(f"Total latency:    {response.total_latency_ms:.1f}ms")
        print(f"Throughput:       {response.tokens_per_second:.2f} tok/s")

        if response.per_hop_latency_ms:
            print(f"Per-hop latency:  {[f'{l:.1f}ms' for l in response.per_hop_latency_ms[:10]]}")

    except grpc.RpcError as e:
        log.error(f"Inference request failed: {e.code()} - {e.details()}")


if __name__ == "__main__":
    main()
