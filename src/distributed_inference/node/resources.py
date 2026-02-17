"""Resource discovery module for node capability detection.

Detects GPU/CPU resources and reports capabilities to the coordinator.
Supports simulated VRAM constraints for local testing.
"""

import platform
from dataclasses import dataclass
from typing import Optional

import psutil
import torch

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class NodeCapabilities:
    """Describes the compute capabilities of a node."""
    vram_mb: int  # Available VRAM in MB (or simulated cap)
    vram_total_mb: int  # Total physical VRAM
    compute_tflops: float  # Estimated compute in TFLOPS
    bandwidth_mbps: float  # Estimated network bandwidth
    effective_bandwidth_mbps: float  # Effective bandwidth for scheduler
    latency_ms: float  # Estimated base network latency to coordinator
    device_type: str  # "cuda" or "cpu"
    device_name: str  # Human-readable device name
    cpu_count: int
    ram_mb: int
    sram_mb: int  # Shared memory / SRAM estimate where available

    def summary(self) -> str:
        """Return a human-readable summary of capabilities."""
        return (
            f"{self.device_name} | "
            f"VRAM: {self.vram_mb}MB / {self.vram_total_mb}MB | "
            f"Compute: {self.compute_tflops:.1f} TFLOPS | "
            f"Net: {self.effective_bandwidth_mbps:.0f}Mbps/{self.latency_ms:.1f}ms | "
            f"Device: {self.device_type}"
        )


def detect_resources(
    max_vram_mb: Optional[int] = None,
    device: str = "auto",
    bandwidth_mbps: Optional[float] = None,
    latency_ms: Optional[float] = None,
) -> NodeCapabilities:
    """Detect available compute resources on this node.

    Args:
        max_vram_mb: Optional artificial VRAM cap in MB for simulation.
            If None, uses all available VRAM.
        device: Device preference: "auto", "cuda", or "cpu".
            "auto" will use CUDA if available, else CPU.

    Returns:
        NodeCapabilities with detected (or simulated) resource info.
    """
    # Determine device
    use_cuda = False
    if device == "auto":
        use_cuda = torch.cuda.is_available()
    elif device == "cuda":
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            log.warning("CUDA requested but not available, falling back to CPU")

    cpu_count = psutil.cpu_count(logical=True)
    ram_mb = psutil.virtual_memory().total // (1024 * 1024)

    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        vram_total_mb = props.total_memory // (1024 * 1024)
        device_name = props.name

        # Estimate TFLOPS from SM count and clock speed
        # Rough estimate: cores * clock * 2 (FMA) / 1e12
        sm_count = props.multi_processor_count
        # Approximate CUDA cores per SM (varies by architecture)
        cores_per_sm = 128  # Rough estimate for modern GPUs
        clock_ghz = props.clock_rate / 1e6  # Convert KHz to GHz
        compute_tflops = (sm_count * cores_per_sm * clock_ghz * 2) / 1e3

        # Apply simulated VRAM cap
        vram_mb = min(vram_total_mb, max_vram_mb) if max_vram_mb else vram_total_mb

        capabilities = NodeCapabilities(
            vram_mb=vram_mb,
            vram_total_mb=vram_total_mb,
            compute_tflops=round(compute_tflops, 2),
            bandwidth_mbps=bandwidth_mbps or 1000.0,
            effective_bandwidth_mbps=bandwidth_mbps or 1000.0,
            latency_ms=latency_ms or 5.0,
            device_type="cuda",
            device_name=device_name,
            cpu_count=cpu_count,
            ram_mb=ram_mb,
            sram_mb=(
                getattr(props, "shared_memory_per_multiprocessor", 0)
                * getattr(props, "multi_processor_count", 0)
            ) // (1024 * 1024),
        )
    else:
        # CPU-only node
        # Use system RAM as "VRAM" equivalent
        vram_total_mb = ram_mb
        vram_mb = min(vram_total_mb, max_vram_mb) if max_vram_mb else vram_total_mb

        capabilities = NodeCapabilities(
            vram_mb=vram_mb,
            vram_total_mb=vram_total_mb,
            compute_tflops=0.1 * cpu_count,  # Rough CPU estimate
            bandwidth_mbps=bandwidth_mbps or 1000.0,
            effective_bandwidth_mbps=bandwidth_mbps or 1000.0,
            latency_ms=latency_ms or 5.0,
            device_type="cpu",
            device_name=f"CPU ({platform.processor() or 'unknown'})",
            cpu_count=cpu_count,
            ram_mb=ram_mb,
            sram_mb=0,
        )

    log.info(f"Detected resources: {capabilities.summary()}")
    if max_vram_mb and max_vram_mb < vram_total_mb:
        log.info(
            f"[yellow]Simulated VRAM cap: {max_vram_mb}MB "
            f"(actual: {vram_total_mb}MB)[/]"
        )

    return capabilities


def get_vram_usage_mb() -> int:
    """Get current GPU VRAM usage in MB.

    Returns 0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) // (1024 * 1024)
    return 0


def get_device(device_type: str) -> torch.device:
    """Get a torch device from a device type string.

    Args:
        device_type: "cuda" or "cpu".

    Returns:
        torch.device instance.
    """
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")
