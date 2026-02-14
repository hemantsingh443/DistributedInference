"""Tests for orchestrator streaming inference events."""

import torch

from distributed_inference.coordinator.orchestrator import Orchestrator
from distributed_inference.coordinator.partitioner import PartitionPlan
from distributed_inference.coordinator.router import HopTrace
from distributed_inference.coordinator.scheduler import ExecutionPlan, PipelineStage


class DummyTokenizer:
    """Tokenizer stub for deterministic streaming tests."""

    eos_token_id = 2

    def __call__(self, prompt, return_tensors="pt"):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            ids = [token_ids]
        else:
            ids = list(token_ids)

        mapping = {1: "Hello", 4: " world", 2: "</s>"}
        pieces = []
        for token_id in ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            pieces.append(mapping.get(token_id, ""))
        return "".join(pieces)


class DummyRouter:
    """Router stub that emits one hop per step with deterministic logits."""

    def __init__(self, stage: PipelineStage):
        self.stage = stage
        self.calls = 0

    def route_forward_stream(self, input_ids, attention_mask, execution_plan, request_id):
        del attention_mask, execution_plan, request_id
        seq_len = input_ids.shape[1]
        logits = torch.zeros((1, seq_len, 8), dtype=torch.float32)

        # Step 0 -> token 4 (" world"), step 1 -> EOS (2).
        target_token = 4 if self.calls == 0 else 2
        logits[0, -1, target_token] = 10.0
        self.calls += 1

        yield HopTrace(
            hop_index=0,
            stage=self.stage,
            output=logits,
            hop_latency_ms=8.0 + self.calls,
        )


def _build_ready_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator()
    stage = PipelineStage(
        node_id="node-1",
        address="localhost:50051",
        start_layer=0,
        end_layer=22,
        has_embedding=True,
        has_lm_head=True,
    )

    orchestrator._model_loaded = True
    orchestrator._partition_plan = PartitionPlan(
        assignments=[],
        model_name="dummy",
        total_layers=22,
    )
    orchestrator._execution_plan = ExecutionPlan(stages=[stage], estimated_total_ms=1.0)
    orchestrator._tokenizer = DummyTokenizer()
    orchestrator.router = DummyRouter(stage)
    return orchestrator


def test_run_inference_stream_emits_hops_tokens_and_completed():
    orchestrator = _build_ready_orchestrator()
    events = list(
        orchestrator.run_inference_stream(
            prompt="test",
            max_tokens=5,
            temperature=0.0,
            request_id="stream-1",
        )
    )

    kinds = [event.WhichOneof("payload") for event in events]
    assert kinds == ["hop", "token", "hop", "completed"]

    hop_event = events[0].hop
    assert hop_event.step == 0
    assert hop_event.node_id == "node-1"
    assert hop_event.start_layer == 0
    assert hop_event.end_layer == 22

    token_event = events[1].token
    assert token_event.step == 0
    assert token_event.token_id == 4
    assert token_event.accumulated_text == "Hello world"

    completed = events[-1].completed
    assert completed.tokens_generated == 1
    assert completed.generated_text == "Hello world"
    assert completed.request_id == "stream-1"


def test_run_inference_consumes_stream_and_returns_final_response():
    orchestrator = _build_ready_orchestrator()
    response = orchestrator.run_inference(
        prompt="test",
        max_tokens=5,
        temperature=0.0,
        request_id="stream-2",
    )

    assert response.request_id == "stream-2"
    assert response.tokens_generated == 1
    assert response.generated_text == "Hello world"
    assert response.total_latency_ms >= 0
    assert len(response.per_hop_latency_ms) == 2
