"""Centralized model weight cache for the coordinator.

Downloads and stores model safetensor files in a local directory so that
nodes can read weights from the shared cache instead of each downloading
independently from HuggingFace Hub.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


def _model_slug(model_name: str) -> str:
    """Convert a model name like 'Qwen/Qwen2-0.5B-Instruct' to a safe directory name."""
    return re.sub(r"[/\\:]", "--", model_name)


class ModelCache:
    """Manages a local cache of model weights for the coordinator.

    Models are stored as ``{cache_dir}/{slug}/`` directories containing
    the full snapshot (config, tokenizer, safetensor shards, index, etc.).
    """

    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        log.info(f"Model cache directory: {self.cache_dir}")

    def get_cache_path(self, model_name: str) -> str:
        """Return the local directory path for a given model."""
        return os.path.join(self.cache_dir, _model_slug(model_name))

    def is_cached(self, model_name: str) -> bool:
        """Check if a model's weights are already fully cached.

        Verifies all shards listed in the safetensors index are present locally.
        For non-sharded models, checks for a single ``model.safetensors``.
        """
        model_dir = self.get_cache_path(model_name)
        if not os.path.isdir(model_dir):
            return False

        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            try:
                import json
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                shard_files = set(weight_map.values())
                for shard in shard_files:
                    if not os.path.exists(os.path.join(model_dir, shard)):
                        return False
                return True
            except Exception as e:
                log.warning(f"Error reading cache index {index_file}: {e}")
                return False

        # Fallback to single-file check
        return os.path.exists(os.path.join(model_dir, "model.safetensors"))

    def ensure_cached(self, model_name: str) -> str:
        """Download model files to the local cache if not already present.

        Uses ``huggingface_hub.snapshot_download`` which handles resume,
        deduplication, and parallel downloads natively.

        Args:
            model_name: HuggingFace Hub model ID (e.g. ``Qwen/Qwen2-0.5B-Instruct``).

        Returns:
            Absolute path to the cached model directory.
        """
        model_dir = self.get_cache_path(model_name)

        if self.is_cached(model_name):
            log.info(f"Model '{model_name}' already cached at {model_dir}")
            return model_dir

        log.info(f"Downloading model '{model_name}' to cache: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)

        # snapshot_download fetches all repo files into a local directory.
        # We exclusively want safetensors, so we ignore other heavy weight formats
        # to save bandwidth and disk space.
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.bin.index.json", "*.pt", "*.h5", "*.msgpack"]
        )

        log.info(f"Model '{model_name}' cached successfully at {model_dir}")
        return model_dir

    def list_cached(self) -> List[str]:
        """List all model names that are fully cached.

        Returns:
            List of model directory names (slugs) present in the cache.
        """
        cached = []
        if not os.path.isdir(self.cache_dir):
            return cached
        for entry in os.listdir(self.cache_dir):
            entry_path = os.path.join(self.cache_dir, entry)
            if os.path.isdir(entry_path):
                has_safetensors = any(
                    f.endswith(".safetensors") for f in os.listdir(entry_path)
                )
                if has_safetensors:
                    cached.append(entry)
        return cached
