"""FastAPI web gateway for distributed inference with SSE streaming."""

import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Iterator, Tuple

import grpc
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from distributed_inference.common.logging import get_logger
from distributed_inference.proto import inference_pb2
from distributed_inference.proto import inference_pb2_grpc

log = get_logger(__name__)


@dataclass
class ManagedNodeProcess:
    """Runtime metadata for a node process managed by the web gateway."""

    node_id: str
    port: int
    coordinator: str
    cmd: list[str]
    process: subprocess.Popen
    created_at: float
    registration_state: str = "pending"  # pending|admitted|rejected
    registration_message: str = ""
    log_tail: Deque[str] = field(default_factory=lambda: deque(maxlen=120))
    lock: Lock = field(default_factory=Lock, repr=False)


class JoinNodeRequest(BaseModel):
    """Payload for dynamically launching a node process."""

    node_id: str | None = None
    coordinator: str | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
    device: str = Field(default="auto", pattern="^(auto|cuda|cpu)$")
    max_vram_mb: int | None = Field(default=None, ge=128)
    bandwidth_mbps: float | None = Field(default=None, gt=0)
    latency_ms: float | None = Field(default=None, gt=0)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")


class CancelRunRequest(BaseModel):
    """Payload for cancelling an active run."""

    coordinator: str | None = None
    reason: str = "cancelled by user"


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
    scheduler_meta = {
        "lane_id": event.lane_id,
        "queue_wait_ms": event.queue_wait_ms,
        "scheduler_retries": event.scheduler_retries,
        "scheduler_policy": event.scheduler_policy,
    }

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
            **scheduler_meta,
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
            **scheduler_meta,
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
            **scheduler_meta,
        }
        return "completed", payload

    payload = {
        "request_id": event.request_id,
        "timestamp_ms": event.timestamp_ms,
        "message": event.error,
        **scheduler_meta,
    }
    return "error", payload


def _is_port_available(port: int) -> bool:
    """Best-effort check whether a localhost TCP port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _coordinator_port_from_addr(address: str) -> int:
    """Extract coordinator port from host:port address, fallback to 50050."""
    try:
        return int(address.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 50050


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate a subprocess best-effort."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _start_node_log_reader(managed: ManagedNodeProcess) -> None:
    """Stream node stdout into tail logs and infer registration outcome."""
    stdout = managed.process.stdout
    if stdout is None:
        return

    def _reader() -> None:
        for raw_line in stdout:
            line = raw_line.strip()
            if not line:
                continue
            with managed.lock:
                managed.log_tail.append(line)
                if (
                    managed.registration_state == "pending"
                    and "Registered successfully" in line
                ):
                    managed.registration_state = "admitted"
                    managed.registration_message = line
                elif "Registration failed:" in line:
                    managed.registration_state = "rejected"
                    managed.registration_message = line.split(
                        "Registration failed:", 1
                    )[-1].strip()

        # Reader EOF: if process exited before registration decision, mark rejected.
        exit_code = managed.process.poll()
        with managed.lock:
            if managed.registration_state == "pending" and exit_code is not None:
                managed.registration_state = "rejected"
                if managed.registration_message:
                    return
                if managed.log_tail:
                    managed.registration_message = managed.log_tail[-1]
                else:
                    managed.registration_message = (
                        f"node process exited with code {exit_code}"
                    )

    thread = threading.Thread(
        target=_reader,
        daemon=True,
        name=f"{managed.node_id}-stdout-reader",
    )
    thread.start()


def _snapshot_managed_node(managed: ManagedNodeProcess) -> dict:
    """Create API payload snapshot for a managed node."""
    exit_code = managed.process.poll()
    with managed.lock:
        registration_state = managed.registration_state
        registration_message = managed.registration_message
        recent_logs = list(managed.log_tail)[-10:]
    return {
        "node_id": managed.node_id,
        "port": managed.port,
        "coordinator": managed.coordinator,
        "pid": managed.process.pid,
        "running": exit_code is None,
        "exit_code": exit_code,
        "created_at": managed.created_at,
        "cmd": " ".join(shlex.quote(token) for token in managed.cmd),
        "registration_state": registration_state,
        "registration_message": registration_message,
        "recent_logs": recent_logs,
    }


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
    app.state.project_root = base_dir.parents[2]
    app.state.managed_nodes: dict[str, ManagedNodeProcess] = {}
    app.state.managed_nodes_lock = Lock()
    app.state.next_node_port = _coordinator_port_from_addr(
        app.state.default_coordinator
    ) + 1

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

    @app.on_event("shutdown")
    async def _terminate_managed_nodes() -> None:
        """Best-effort cleanup for node processes launched by the web UI/API."""
        with app.state.managed_nodes_lock:
            managed_nodes = list(app.state.managed_nodes.values())
            app.state.managed_nodes.clear()
        for managed in managed_nodes:
            _stop_process(managed.process)

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
        with request.app.state.managed_nodes_lock:
            managed_nodes = len(request.app.state.managed_nodes)
        return {
            "status": "ok",
            "coordinator": request.app.state.default_coordinator,
            "managed_nodes": managed_nodes,
        }

    @app.get("/api/health-fragment", response_class=HTMLResponse)
    def health_fragment(request: Request):
        coordinator = request.app.state.default_coordinator
        with request.app.state.managed_nodes_lock:
            managed_nodes = len(request.app.state.managed_nodes)
        return (
            "<span class='health-dot'></span>"
            "<span>Web ready</span>"
            f"<span class='muted'>Coordinator: {coordinator}</span>"
            f"<span class='muted'>Managed nodes: {managed_nodes}</span>"
        )

    @app.get("/api/runs/{request_id}")
    def get_run_events(request_id: str, request: Request):
        events = request.app.state.logs.get(request_id)
        if not events:
            raise HTTPException(status_code=404, detail="request_id not found")
        return JSONResponse({"request_id": request_id, "events": events})

    @app.post("/api/runs/{request_id}/cancel")
    def cancel_run(request_id: str, payload: CancelRunRequest, request: Request):
        coordinator_addr = payload.coordinator or request.app.state.default_coordinator
        options = [
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(coordinator_addr, options=options)
        stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
        try:
            response = stub.CancelInference(
                inference_pb2.CancelInferenceRequest(
                    request_id=request_id,
                    reason=payload.reason,
                ),
                timeout=10,
            )
            return JSONResponse(
                {
                    "request_id": request_id,
                    "accepted": bool(response.accepted),
                    "status": response.status,
                }
            )
        except grpc.RpcError as rpc_error:
            raise HTTPException(
                status_code=502,
                detail=f"{rpc_error.code().name}: {rpc_error.details()}",
            ) from rpc_error
        finally:
            channel.close()

    @app.get("/api/nodes")
    def list_managed_nodes(request: Request):
        """List node processes launched via web controls."""
        with request.app.state.managed_nodes_lock:
            nodes = [
                _snapshot_managed_node(managed)
                for managed in request.app.state.managed_nodes.values()
            ]
        nodes.sort(key=lambda item: item["created_at"], reverse=True)
        return JSONResponse({"nodes": nodes})

    @app.post("/api/nodes/join")
    def join_node(request: Request, payload: JoinNodeRequest):
        """Spawn a new node process with caller-provided resource profile."""
        with request.app.state.managed_nodes_lock:
            node_id = payload.node_id or f"web-node-{uuid.uuid4().hex[:8]}"
            existing = request.app.state.managed_nodes.get(node_id)
            if existing and existing.process.poll() is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"node_id '{node_id}' is already running",
                )

            port = payload.port
            if port is None:
                cursor = max(request.app.state.next_node_port, 1024)
                while not _is_port_available(cursor):
                    cursor += 1
                port = cursor
                request.app.state.next_node_port = cursor + 1
            elif not _is_port_available(port):
                raise HTTPException(
                    status_code=409,
                    detail=f"Port {port} is already in use",
                )

            coordinator = payload.coordinator or request.app.state.default_coordinator
            cmd = [
                sys.executable,
                "-m",
                "distributed_inference.cli.start_node",
                "--port",
                str(port),
                "--coordinator",
                coordinator,
                "--device",
                payload.device,
                "--node-id",
                node_id,
                "--require-registration",
                "--log-level",
                payload.log_level,
            ]
            if payload.max_vram_mb is not None:
                cmd.extend(["--max-vram-mb", str(payload.max_vram_mb)])
            if payload.bandwidth_mbps is not None:
                cmd.extend(["--bandwidth-mbps", str(payload.bandwidth_mbps)])
            if payload.latency_ms is not None:
                cmd.extend(["--latency-ms", str(payload.latency_ms)])

            env = dict(os.environ)
            src_path = str(request.app.state.project_root / "src")
            existing_pythonpath = env.get("PYTHONPATH")
            if existing_pythonpath:
                env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}"
            else:
                env["PYTHONPATH"] = src_path

            process = subprocess.Popen(
                cmd,
                cwd=str(request.app.state.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            managed = ManagedNodeProcess(
                node_id=node_id,
                port=port,
                coordinator=coordinator,
                cmd=cmd,
                process=process,
                created_at=time.time(),
            )
            _start_node_log_reader(managed)
            request.app.state.managed_nodes[node_id] = managed

        deadline = time.time() + 12.0
        while time.time() < deadline:
            snapshot = _snapshot_managed_node(managed)
            if snapshot["registration_state"] == "admitted":
                return JSONResponse(
                    {
                        "success": True,
                        "node_id": node_id,
                        "port": port,
                        "coordinator": coordinator,
                        "pid": process.pid,
                        "cmd": " ".join(shlex.quote(token) for token in cmd),
                        "admitted": True,
                        "registration_state": "admitted",
                        "registration_message": snapshot["registration_message"],
                    }
                )
            if snapshot["registration_state"] == "rejected":
                _stop_process(process)
                with request.app.state.managed_nodes_lock:
                    request.app.state.managed_nodes.pop(node_id, None)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        snapshot["registration_message"]
                        or "Node rejected by coordinator admission policy"
                    ),
                )
            if snapshot["exit_code"] is not None:
                with request.app.state.managed_nodes_lock:
                    request.app.state.managed_nodes.pop(node_id, None)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        snapshot["registration_message"]
                        or f"Node exited early with code {snapshot['exit_code']}"
                    ),
                )
            time.sleep(0.1)

        snapshot = _snapshot_managed_node(managed)
        return JSONResponse(
            {
                "success": True,
                "node_id": node_id,
                "port": port,
                "coordinator": coordinator,
                "pid": process.pid,
                "cmd": " ".join(shlex.quote(token) for token in cmd),
                "admitted": False,
                "registration_state": snapshot["registration_state"],
                "registration_message": snapshot["registration_message"],
            }
        )

    @app.post("/api/nodes/{node_id}/stop")
    def stop_node(request: Request, node_id: str):
        """Stop a node process launched by this web process."""
        with request.app.state.managed_nodes_lock:
            managed = request.app.state.managed_nodes.get(node_id)
            if not managed:
                raise HTTPException(
                    status_code=404,
                    detail=f"Managed node '{node_id}' not found",
                )

        process = managed.process
        _stop_process(process)

        with request.app.state.managed_nodes_lock:
            request.app.state.managed_nodes.pop(node_id, None)

        return JSONResponse(
            {
                "success": True,
                "node_id": node_id,
                "exit_code": process.poll(),
            }
        )

    @app.post("/api/nodes/{node_id}/remove")
    def remove_node(request: Request, node_id: str):
        """Remove a non-running managed node entry from the web registry."""
        with request.app.state.managed_nodes_lock:
            managed = request.app.state.managed_nodes.get(node_id)
            if not managed:
                raise HTTPException(
                    status_code=404,
                    detail=f"Managed node '{node_id}' not found",
                )
            if managed.process.poll() is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Managed node '{node_id}' is still running; stop it first",
                )
            request.app.state.managed_nodes.pop(node_id, None)

        return JSONResponse({"success": True, "node_id": node_id})

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
        user_id: str | None = Query(None),
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
                user_id=user_id or "",
            )

            try:
                for event in stub.SubmitInferenceStream(grpc_request, timeout=3600):
                    event_type, payload = _event_to_payload(event)
                    request.app.state.logs.append(
                        request_id, {"type": event_type, **payload}
                    )
                    sse_event = (
                        "inference_error" if event_type == "error" else event_type
                    )
                    yield _sse(sse_event, payload)

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
                yield _sse("inference_error", payload)
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
