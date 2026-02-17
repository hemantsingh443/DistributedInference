"""Tests for node admission policy."""

from distributed_inference.common.config import CoordinatorConfig
from distributed_inference.coordinator.admission import AdmissionController


def test_admission_accepts_threshold_match() -> None:
    config = CoordinatorConfig(
        min_vram_mb=1024,
        min_compute_tflops=2.5,
        gpu_required=False,
    )
    admission = AdmissionController(config)

    decision = admission.evaluate(
        vram_mb=1024,
        compute_tflops=2.5,
        device_type="cpu",
    )
    assert decision.admitted is True
    assert decision.reason == ""


def test_admission_rejects_low_vram() -> None:
    config = CoordinatorConfig(min_vram_mb=2048, min_compute_tflops=0.0)
    admission = AdmissionController(config)

    decision = admission.evaluate(
        vram_mb=1024,
        compute_tflops=10.0,
        device_type="cuda",
    )
    assert decision.admitted is False
    assert "Insufficient VRAM" in decision.reason


def test_admission_rejects_when_gpu_required() -> None:
    config = CoordinatorConfig(
        min_vram_mb=512,
        min_compute_tflops=0.1,
        gpu_required=True,
    )
    admission = AdmissionController(config)

    decision = admission.evaluate(
        vram_mb=4096,
        compute_tflops=8.0,
        device_type="cpu",
    )
    assert decision.admitted is False
    assert "GPU is required" in decision.reason
