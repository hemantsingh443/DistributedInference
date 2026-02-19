"""Tests for orchestrator rebalance failure handling."""

from distributed_inference.common.config import SystemConfig
from distributed_inference.coordinator.orchestrator import NodeLoadError, Orchestrator
from distributed_inference.coordinator.partitioner import LayerAssignment, PartitionPlan
from distributed_inference.coordinator.registry import NodeState
from distributed_inference.coordinator.scheduler import ExecutionPlan


def test_rebalance_load_failure_keeps_viable_old_plan(monkeypatch):
    config = SystemConfig()
    orchestrator = Orchestrator(config=config)
    orchestrator._running = True
    orchestrator._tokenizer = object()  # Avoid tokenizer download path.

    node = orchestrator.registry.register(
        node_id="node-1",
        address="localhost:50051",
        vram_mb=2048,
        compute_tflops=5.0,
        bandwidth_mbps=1000.0,
        device_type="cpu",
        device_name="test",
        admitted=True,
    )

    old_plan = PartitionPlan(
        assignments=[LayerAssignment("node-1", 0, 22, has_embedding=True, has_lm_head=True)],
        model_name="test-model",
        total_layers=22,
    )
    old_exec = ExecutionPlan(stages=[], estimated_total_ms=1.0)
    orchestrator._partition_plan = old_plan
    orchestrator._execution_plan = old_exec
    orchestrator._model_loaded = True

    new_plan = PartitionPlan(
        assignments=[LayerAssignment("node-1", 0, 22, has_embedding=True, has_lm_head=True)],
        model_name="test-model",
        total_layers=22,
    )
    new_exec = ExecutionPlan(stages=[], estimated_total_ms=2.0)

    monkeypatch.setattr(orchestrator, "_wait_for_drain", lambda timeout_sec: True)
    monkeypatch.setattr(orchestrator, "_get_rebalance_candidate_nodes", lambda: [node])
    monkeypatch.setattr(orchestrator.router, "check_node_health", lambda address: object())
    monkeypatch.setattr(
        "distributed_inference.coordinator.orchestrator.partition_model",
        lambda **kwargs: new_plan,
    )
    monkeypatch.setattr(orchestrator.scheduler, "create_execution_plan", lambda plan: new_exec)
    monkeypatch.setattr(
        orchestrator,
        "_load_partition_plan",
        lambda plan: (_ for _ in ()).throw(
            NodeLoadError("node-1", "simulated shard load failure")
        ),
    )

    ok = orchestrator._rebalance_pipeline("topology:register")
    assert ok is False
    assert orchestrator._partition_plan is old_plan
    assert orchestrator._execution_plan is old_exec
    assert orchestrator._model_loaded is True


def test_resource_load_error_sets_backoff_without_marking_dead(monkeypatch):
    orchestrator = Orchestrator()
    node = orchestrator.registry.register(
        node_id="node-1",
        address="localhost:50051",
        vram_mb=2048,
        compute_tflops=5.0,
        bandwidth_mbps=1000.0,
        device_type="cpu",
        device_name="test",
        admitted=True,
    )
    orchestrator.config.coordinator.node_load_failure_backoff_sec = 10.0
    orchestrator._handle_node_load_error(
        NodeLoadError("node-1", "Shard load failed: paging file is too small")
    )
    updated = orchestrator.registry.get_node("node-1")
    assert updated is not None
    assert updated.state == NodeState.IDLE
    assert "node-1" in orchestrator._node_load_backoff_until

    monkeypatch.setattr(orchestrator.router, "check_node_health", lambda address: object())
    candidates = orchestrator._get_rebalance_candidate_nodes()
    assert candidates == []


def test_load_partition_plan_skips_unchanged_assignment(monkeypatch):
    orchestrator = Orchestrator()
    node = orchestrator.registry.register(
        node_id="node-1",
        address="localhost:50051",
        vram_mb=2048,
        compute_tflops=5.0,
        bandwidth_mbps=1000.0,
        device_type="cpu",
        device_name="test",
        admitted=True,
    )
    orchestrator.registry.set_node_assignment(
        node_id="node-1",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )

    plan = PartitionPlan(
        assignments=[LayerAssignment("node-1", 0, 22, has_embedding=True, has_lm_head=True)],
        model_name="test-model",
        total_layers=22,
    )

    calls = {"loads": 0}
    monkeypatch.setattr(orchestrator.router, "check_node_health", lambda address: object())
    monkeypatch.setattr(
        orchestrator.router,
        "load_shard_on_node",
        lambda **kwargs: calls.__setitem__("loads", calls["loads"] + 1),
    )

    orchestrator._load_partition_plan(plan)
    assert calls["loads"] == 0
