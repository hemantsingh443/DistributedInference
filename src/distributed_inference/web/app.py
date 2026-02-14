"""FastAPI web gateway for distributed inference with SSE streaming."""

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Iterator, Tuple

import grpc
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from distributed_inference.common.logging import get_logger
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


class RunLogStore:
    """In-memory request-scoped event store with a fixed per-run cap."""

    def __init__(self, max_events_per_run: int = 5000):
        self._max_events_per_run = max_events_per_run
        self._events: Dict[str, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=self._max_events_per_run)
        )
        self._lock = Lock()

    def append(self, request_id: str, event: dict) -> None:
        with self._lock:
            self._events[request_id].append(event)

    def get(self, request_id: str) -> list[dict]:
        with self._lock:
            if request_id not in self._events:
                return []
            return list(self._events[request_id])


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _event_to_payload(event: inference_pb2.InferenceEvent) -> Tuple[str, dict]:
    kind = event.WhichOneof("payload")

    if kind == "hop":
        payload = {
            "request_id": event.request_id,
            "timestamp_ms": event.timestamp_ms,
            "step": event.hop.step,
            "hop_index": event.hop.hop_index,
            "node_id": event.hop.node_id,
            "address": event.hop.address,
            "start_layer": event.hop.start_layer,
            "end_layer": event.hop.end_layer,
            "hop_latency_ms": event.hop.hop_latency_ms,
        }
        return "hop", payload

    if kind == "token":
        payload = {
            "request_id": event.request_id,
            "timestamp_ms": event.timestamp_ms,
            "step": event.token.step,
            "token_id": event.token.token_id,
            "token_text": event.token.token_text,
            "accumulated_text": event.token.accumulated_text,
        }
        return "token", payload

    if kind == "completed":
        payload = {
            "request_id": event.completed.request_id,
            "timestamp_ms": event.timestamp_ms,
            "generated_text": event.completed.generated_text,
            "tokens_generated": event.completed.tokens_generated,
            "total_latency_ms": event.completed.total_latency_ms,
            "tokens_per_second": event.completed.tokens_per_second,
            "per_hop_latency_ms": list(event.completed.per_hop_latency_ms),
        }
        return "completed", payload

    payload = {
        "request_id": event.request_id,
        "timestamp_ms": event.timestamp_ms,
        "message": event.error,
    }
    return "error", payload


def create_app(default_coordinator: str | None = None) -> FastAPI:
    """Create the FastAPI app instance."""
    base_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))

    app = FastAPI(title="Distributed Inference Web Gateway")
    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

    app.state.default_coordinator = (
        default_coordinator
        or os.getenv("DI_COORDINATOR_ADDRESS")
        or "localhost:50050"
    )
    app.state.logs = RunLogStore(max_events_per_run=5000)

    @app.on_event("startup")
    async def _install_exception_filter() -> None:
        """Suppress noisy expected Windows SSE disconnect reset errors."""
        loop = asyncio.get_running_loop()
        default_handler = loop.get_exception_handler()

        def _handler(event_loop, context):
            exc = context.get("exception")
            message = context.get("message", "")

            # Expected on Windows Proactor when SSE client closes connection.
            if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
                if "_ProactorBasePipeTransport._call_connection_lost" in message:
                    return

            if default_handler is not None:
                default_handler(event_loop, context)
            else:
                event_loop.default_exception_handler(context)

        loop.set_exception_handler(_handler)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_coordinator": request.app.state.default_coordinator,
            },
        )

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ok",
            "coordinator": request.app.state.default_coordinator,
        }

    @app.get("/api/health-fragment", response_class=HTMLResponse)
    def health_fragment(request: Request):
        coordinator = request.app.state.default_coordinator
        return (
            "<span class='health-dot'></span>"
            "<span>Web ready</span>"
            f"<span class='muted'>Coordinator: {coordinator}</span>"
        )

    @app.get("/api/runs/{request_id}")
    def get_run_events(request_id: str, request: Request):
        events = request.app.state.logs.get(request_id)
        if not events:
            raise HTTPException(status_code=404, detail="request_id not found")
        return JSONResponse({"request_id": request_id, "events": events})

    @app.get("/api/stream")
    def stream_inference(
        request: Request,
        prompt: str = Query(..., min_length=1),
        max_tokens: int = Query(50, ge=1, le=512),
        temperature: float = Query(0.7, ge=0.0, le=2.0),
        top_p: float = Query(0.9, ge=0.0, le=1.0),
        top_k: int = Query(50, ge=0, le=2000),
        coordinator: str | None = Query(None),
        request_id: str | None = Query(None),
    ):
        request_id = request_id or uuid.uuid4().hex[:8]
        coordinator_addr = coordinator or request.app.state.default_coordinator

        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        options = [
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]

        def event_stream() -> Iterator[str]:
            channel = grpc.insecure_channel(coordinator_addr, options=options)
            stub = inference_pb2_grpc.CoordinatorServiceStub(channel)

            start_payload = {
                "request_id": request_id,
                "coordinator": coordinator_addr,
                "timestamp_ms": int(time.time() * 1000),
            }
            request.app.state.logs.append(request_id, {"type": "start", **start_payload})
            yield _sse("start", start_payload)

            grpc_request = inference_pb2.InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            try:
                for event in stub.SubmitInferenceStream(grpc_request, timeout=3600):
                    event_type, payload = _event_to_payload(event)
                    request.app.state.logs.append(
                        request_id, {"type": event_type, **payload}
                    )
                    yield _sse(event_type, payload)

            except grpc.RpcError as rpc_error:
                payload = {
                    "request_id": request_id,
                    "timestamp_ms": int(time.time() * 1000),
                    "message": f"{rpc_error.code().name}: {rpc_error.details()}",
                }
                log.error(
                    f"Streaming RPC failed for {request_id}: "
                    f"{rpc_error.code()} {rpc_error.details()}"
                )
                request.app.state.logs.append(request_id, {"type": "error", **payload})
                yield _sse("error", payload)
            finally:
                channel.close()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    return app


app = create_app()
