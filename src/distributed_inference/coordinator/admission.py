"""Admission controller for dynamic node onboarding."""

from dataclasses import dataclass

from distributed_inference.common.config import CoordinatorConfig


@dataclass(frozen=True)
class AdmissionDecision:
    """Admission evaluation result for a node registration attempt."""

    admitted: bool
    reason: str = ""


class AdmissionController:
    """Evaluates whether a node can join the inference worker pool."""

    def __init__(self, config: CoordinatorConfig):
        self.config = config

    def evaluate(
        self,
        *,
        vram_mb: int,
        compute_tflops: float,
        device_type: str,
    ) -> AdmissionDecision:
        """Evaluate admission against configured minimum thresholds."""
        if self.config.gpu_required and device_type != "cuda":
            return AdmissionDecision(
                admitted=False,
                reason="GPU is required by coordinator policy",
            )

        if vram_mb < self.config.min_vram_mb:
            return AdmissionDecision(
                admitted=False,
                reason=(
                    f"Insufficient VRAM: {vram_mb}MB "
                    f"< minimum {self.config.min_vram_mb}MB"
                ),
            )

        if compute_tflops < self.config.min_compute_tflops:
            return AdmissionDecision(
                admitted=False,
                reason=(
                    f"Insufficient compute: {compute_tflops:.2f} TFLOPS "
                    f"< minimum {self.config.min_compute_tflops:.2f} TFLOPS"
                ),
            )

        return AdmissionDecision(admitted=True)
