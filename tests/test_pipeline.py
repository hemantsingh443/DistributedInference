"""Integration tests for the distributed inference pipeline.

These tests require model downloads and are marked as slow.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import torch


class TestPipelineBasic:
    """Basic pipeline tests that don't require model downloads."""

    def test_config_loading(self):
        """Test that default config loads correctly."""
        from distributed_inference.common.config import load_config
        config = load_config()
        assert config.model.name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.model.num_layers == 22
        assert config.coordinator.port == 50050

    def test_profiler_metrics(self):
        """Test profiler records and aggregates correctly."""
        from distributed_inference.benchmarks.profiler import InferenceProfiler

        profiler = InferenceProfiler()
        profiler.record_run(
            request_id="test-1",
            prompt_tokens=10,
            generated_tokens=20,
            total_latency_ms=1000.0,
            per_hop_latencies=[200.0, 300.0, 500.0],
            node_ids=["node-1", "node-2", "node-3"],
        )

        summary = profiler.get_summary()
        assert summary["runs"] == 1
        assert summary["avg_latency_ms"] == 1000.0
        assert summary["total_tokens_generated"] == 20

    def test_profiler_comparison_report(self):
        """Test profiler comparison output."""
        from distributed_inference.benchmarks.profiler import InferenceProfiler

        profiler = InferenceProfiler()
        profiler.record_run("t1", 10, 20, 1000.0, [500.0, 500.0])

        report = profiler.print_comparison(single_node_latency_ms=800.0)
        assert "Benchmark" in report
        assert "Overhead" in report

    def test_scheduler_execution_plan(self):
        """Test scheduler creates valid execution plans."""
        from distributed_inference.coordinator.registry import NodeRegistry
        from distributed_inference.coordinator.scheduler import Scheduler
        from distributed_inference.coordinator.partitioner import (
            PartitionPlan, LayerAssignment,
        )

        registry = NodeRegistry()
        registry.register("node-1", "localhost:50051", 2048)
        registry.register("node-2", "localhost:50052", 1024)

        plan = PartitionPlan(
            assignments=[
                LayerAssignment("node-1", 0, 14, has_embedding=True),
                LayerAssignment("node-2", 14, 22, has_lm_head=True),
            ],
            model_name="test",
            total_layers=22,
        )

        scheduler = Scheduler(registry=registry)
        exec_plan = scheduler.create_execution_plan(plan)

        assert exec_plan.num_stages == 2
        assert exec_plan.stages[0].start_layer == 0
        assert exec_plan.stages[1].start_layer == 14

    def test_scheduler_throughput_estimate(self):
        """Test scheduler throughput estimation."""
        from distributed_inference.coordinator.registry import NodeRegistry
        from distributed_inference.coordinator.scheduler import Scheduler
        from distributed_inference.coordinator.partitioner import (
            PartitionPlan, LayerAssignment,
        )

        registry = NodeRegistry()
        registry.register("node-1", "localhost:50051", 2048)

        plan = PartitionPlan(
            assignments=[
                LayerAssignment("node-1", 0, 22, has_embedding=True, has_lm_head=True),
            ],
            model_name="test",
            total_layers=22,
        )

        scheduler = Scheduler(registry=registry)
        exec_plan = scheduler.create_execution_plan(plan)
        metrics = scheduler.estimate_throughput(exec_plan)

        assert metrics["total_latency_ms"] > 0
        assert metrics["estimated_tokens_per_second"] > 0
