"""Model shard executor for running forward passes on assigned layers.

Loads specific transformer layers from a HuggingFace model and runs
inference on them. Handles both first-node (embedding) and last-node
(lm_head) special components.

Uses selective weight loading via safetensors to avoid loading the
entire model into memory. Each node only downloads and materializes
the exact layer weights it needs.

Designed for transformers 5.x which requires using the model's own
forward() method rather than calling individual layers directly.
"""

from collections import OrderedDict
from dataclasses import dataclass
import gc
import json
import os
import re
import threading
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoModelForCausalLM, AutoConfig

from distributed_inference.common.logging import get_logger
from distributed_inference.node.resources import get_device, get_vram_usage_mb

log = get_logger(__name__)


@dataclass
class CacheEntry:
    """Per-request KV cache metadata."""
    past_key_values: Any
    tokens_seen: int
    last_access_ts: float


class ShardExecutor:
    """Executes forward passes on a subset of model layers.

    Loads specific transformer layers from a pre-trained model and runs
    input tensors through them. Uses selective weight loading via
    safetensors so each node only materializes its assigned layers,
    avoiding full-model memory spikes.

    Uses the model's own forward() method to ensure compatibility with
    transformers 5.x internal masking and positional embedding logic.
    """

    def __init__(
        self,
        device_type: str = "cpu",
        max_cached_requests: int = 1,
        max_cache_tokens_per_request: int = 4096,
    ):
        self.device = get_device(device_type)
        self.model = None            # LlamaModel (the inner model)
        self.lm_head = None          # Linear projection to vocab
        self.model_config = None
        self.max_cached_requests = max(1, int(max_cached_requests))
        self.max_cache_tokens_per_request = max(1, int(max_cache_tokens_per_request))
        self._kv_cache_by_request: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()

        self.start_layer: int = 0
        self.end_layer: int = 0
        self.has_embedding: bool = False
        self.has_lm_head: bool = False
        self._loaded: bool = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _get_shard_weight_keys(
        start_layer: int,
        end_layer: int,
        has_embedding: bool,
        has_lm_head: bool,
    ) -> set[str]:
        """Build the set of weight-key prefixes this shard needs."""
        prefixes: set[str] = set()
        for i in range(start_layer, end_layer):
            prefixes.add(f"model.layers.{i}.")
        if has_embedding:
            prefixes.add("model.embed_tokens.")
        if has_lm_head:
            prefixes.add("lm_head.")
            prefixes.add("model.norm.")
        return prefixes

    @staticmethod
    def _remap_layer_key(key: str, start_layer: int) -> str:
        """Remap a full-model weight key to local shard indices.

        E.g. ``model.layers.15.self_attn.q_proj.weight`` with
        start_layer=15 becomes ``model.layers.0.self_attn.q_proj.weight``.
        """
        m = re.match(r"^(model\.layers\.)(\d+)(\..*)", key)
        if m:
            original_idx = int(m.group(2))
            local_idx = original_idx - start_layer
            return f"{m.group(1)}{local_idx}{m.group(3)}"
        return key

    def _resolve_safetensor_files(
        self,
        model_name: str,
        needed_prefixes: set[str],
    ) -> dict[str, dict[str, str]]:
        """Determine which safetensor files contain needed weights.

        Returns:
            Dict mapping local file path -> {weight_key: ...} for each
            key that matches a needed prefix.  The value dict maps
            original weight key -> remapped weight key.
        """
        # Try sharded index first
        try:
            index_path = hf_hub_download(
                model_name, "model.safetensors.index.json"
            )
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map: dict[str, str] = index_data["weight_map"]
        except Exception:
            # Single-file model — all weights live in one file
            single_path = hf_hub_download(model_name, "model.safetensors")
            weight_map = None  # handled below

        if weight_map is None:
            # Single file: return all matching keys from the single file
            return {single_path: {}}  # empty dict → load all & filter later

        # Sharded model: group needed keys by shard file
        shard_files_needed: dict[str, dict[str, str]] = {}
        repo_dir = os.path.dirname(index_path)
        for weight_key, shard_filename in weight_map.items():
            if any(weight_key.startswith(p) for p in needed_prefixes):
                shard_path = os.path.join(repo_dir, shard_filename)
                if not os.path.exists(shard_path):
                    shard_path = hf_hub_download(model_name, shard_filename)
                if shard_path not in shard_files_needed:
                    shard_files_needed[shard_path] = {}
                shard_files_needed[shard_path][weight_key] = weight_key
        return shard_files_needed

    @torch.no_grad()
    def load_shard(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        dtype: str = "float16",
    ) -> dict:
        """Load specific layers from a pre-trained model.

        Uses selective weight loading via safetensors to avoid
        materializing the full model. Creates an empty model shell
        with only the needed layers, then loads matching weights from
        the safetensor files with layer-index remapping.

        Args:
            model_name: HuggingFace model name/path.
            start_layer: First layer index (inclusive).
            end_layer: Last layer index (exclusive).
            has_embedding: Whether this node holds the embedding layer.
            has_lm_head: Whether this node holds the output head.
            dtype: Data type for weights ("float16" or "float32").

        Returns:
            Dict with load stats (vram_used_mb, num_layers, load_time_ms).
        """
        start_time = time.time()

        self.start_layer = start_layer
        self.end_layer = end_layer
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head

        log.info(
            f"Loading shard: layers [{start_layer}, {end_layer}) "
            f"embed={has_embedding} lm_head={has_lm_head} "
            f"dtype={dtype} device={self.device}"
        )

        # Free previously loaded shard before constructing a new model to
        # avoid transient peak-memory spikes during rebalance/reload.
        if self._loaded:
            self.unload()
            gc.collect()

        # Load model config and create an empty model shell with only
        # the number of layers this shard needs.
        self.model_config = AutoConfig.from_pretrained(model_name)
        num_layers = end_layer - start_layer

        original_num_layers = self.model_config.num_hidden_layers
        self.model_config.num_hidden_layers = num_layers
        self.model_config.use_cache = True

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        # Construct empty model on CPU with meta weights to minimize memory
        shell_model = AutoModelForCausalLM.from_config(
            self.model_config,
            torch_dtype=torch_dtype,
        )

        # Determine which weight prefixes this shard needs
        needed_prefixes = self._get_shard_weight_keys(
            start_layer, end_layer, has_embedding, has_lm_head
        )

        # Resolve which safetensor files contain the needed weights
        log.info(f"Resolving safetensor files for {len(needed_prefixes)} weight groups")
        shard_file_map = self._resolve_safetensor_files(model_name, needed_prefixes)

        # Load and remap weights from safetensor files
        state_dict: dict[str, torch.Tensor] = {}
        for shard_path, key_mapping in shard_file_map.items():
            file_tensors = safetensors_load_file(shard_path, device="cpu")
            for orig_key, tensor in file_tensors.items():
                if any(orig_key.startswith(p) for p in needed_prefixes):
                    remapped_key = self._remap_layer_key(orig_key, start_layer)
                    state_dict[remapped_key] = tensor.to(torch_dtype)
            del file_tensors

        # Load the selective state dict into the model shell
        missing, unexpected = shell_model.load_state_dict(state_dict, strict=False)
        del state_dict

        # Log any issues — some missing keys are expected for non-embed/non-lm_head nodes
        if unexpected:
            log.warning(f"Unexpected keys during shard load: {unexpected[:5]}")
        expected_missing = set()
        if not has_embedding:
            expected_missing.add("model.embed_tokens.weight")
        if not has_lm_head:
            expected_missing.add("lm_head.weight")
            expected_missing.add("model.norm.weight")
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            log.warning(
                f"Missing {len(real_missing)} weight keys during shard load "
                f"(first 5): {real_missing[:5]}"
            )

        # Reindex layer ids for cache slot alignment
        self._renumber_layer_indices(shell_model.model.layers)

        # Store the inner model (LlamaModel) and optionally lm_head
        self.model = shell_model.model

        if has_lm_head:
            self.lm_head = shell_model.lm_head
        else:
            # Intermediate nodes: skip final RMSNorm (only last node applies it)
            self.model.norm = nn.Identity()
            self.lm_head = None

        # Move to target device
        self.model = self.model.to(self.device)
        if self.lm_head is not None:
            self.lm_head = self.lm_head.to(self.device)

        # Free the wrapper
        del shell_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = True
        self.clear_all_cache()
        load_time_ms = (time.time() - start_time) * 1000
        vram_used = get_vram_usage_mb()

        log.info(
            f"Shard loaded: {num_layers} layers in {load_time_ms:.0f}ms, "
            f"VRAM used: {vram_used}MB"
        )

        return {
            "vram_used_mb": vram_used,
            "num_layers": num_layers,
            "load_time_ms": load_time_ms,
        }

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        request_id: str = "",
        use_cache: bool = False,
        reset_cache: bool = False,
        cache_position: Optional[int] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Run forward pass through the loaded layers.

        Args:
            hidden_states: Input tensor. If this node has the embedding layer,
                this should be input_ids (LongTensor). Otherwise, it's the
                hidden states from the previous node.
            attention_mask: Attention mask tensor.
            position_ids: Position IDs tensor.
            request_id: Unique request ID used to isolate cache state.
            use_cache: Whether to use/update KV cache.
            reset_cache: Whether to clear any existing cache for request_id.
            cache_position: Absolute token position for this input chunk.
            is_prefill: Whether this call is prompt prefill vs decode.

        Returns:
            Output hidden states (or logits if this node has lm_head).
        """
        if not self._loaded:
            raise RuntimeError("No shard loaded. Call load_shard() first.")
        if use_cache and not request_id:
            raise RuntimeError("request_id is required when use_cache=True")
        if reset_cache and request_id:
            self.clear_request_cache(request_id)

        # Move inputs to device
        hidden_states = hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        if not use_cache:
            return self._run_forward_no_cache(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        if is_prefill:
            return self._run_forward_prefill(
                request_id=request_id,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )

        return self._run_forward_decode(
            request_id=request_id,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )

    def _run_forward_no_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = self._invoke_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            past_key_values=None,
            cache_position_tensor=None,
        )
        return out.last_hidden_state

    def _run_forward_prefill(
        self,
        request_id: str,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cache_position: Optional[int],
    ) -> torch.Tensor:
        seq_len = self._sequence_length(hidden_states)
        if seq_len <= 0:
            raise RuntimeError("Prefill input sequence length must be > 0")

        if seq_len > self.max_cache_tokens_per_request:
            raise RuntimeError(
                f"Prefill sequence length {seq_len} exceeds max cache tokens "
                f"{self.max_cache_tokens_per_request}"
            )

        cache_pos_tensor = self._build_cache_position(
            seq_len=seq_len,
            last_cache_position=cache_position,
            expected_start=0,
        )
        out = self._invoke_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=None,
            cache_position_tensor=cache_pos_tensor,
        )

        self._store_cache_entry(
            request_id=request_id,
            past_key_values=out.past_key_values,
            tokens_seen=int(cache_pos_tensor[-1].item()) + 1,
        )
        return out.last_hidden_state

    def _run_forward_decode(
        self,
        request_id: str,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cache_position: Optional[int],
    ) -> torch.Tensor:
        with self._cache_lock:
            entry = self._kv_cache_by_request.get(request_id)
        if entry is None:
            raise RuntimeError(
                f"KV cache miss for request {request_id}. Prefill must run first."
            )

        seq_len = self._sequence_length(hidden_states)
        if seq_len <= 0:
            raise RuntimeError("Decode input sequence length must be > 0")

        cache_pos_tensor = self._build_cache_position(
            seq_len=seq_len,
            last_cache_position=cache_position,
            expected_start=entry.tokens_seen,
        )

        decode_start = int(cache_pos_tensor[0].item())
        if decode_start != entry.tokens_seen:
            raise RuntimeError(
                f"Cache position mismatch for request {request_id}: "
                f"expected {entry.tokens_seen}, got {decode_start}"
            )

        tokens_seen = int(cache_pos_tensor[-1].item()) + 1
        if tokens_seen > self.max_cache_tokens_per_request:
            raise RuntimeError(
                f"Cache tokens {tokens_seen} exceed limit "
                f"{self.max_cache_tokens_per_request} for {request_id}"
            )

        out = self._invoke_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=entry.past_key_values,
            cache_position_tensor=cache_pos_tensor,
        )

        self._store_cache_entry(
            request_id=request_id,
            past_key_values=out.past_key_values,
            tokens_seen=tokens_seen,
        )
        return out.last_hidden_state

    def _invoke_model(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        use_cache: bool,
        past_key_values: Optional[Any],
        cache_position_tensor: Optional[torch.Tensor],
    ):
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if cache_position_tensor is not None:
            kwargs["cache_position"] = cache_position_tensor

        if self.has_embedding:
            out = self.model(input_ids=hidden_states, **kwargs)
        else:
            out = self.model(inputs_embeds=hidden_states, **kwargs)

        hidden = out.last_hidden_state
        if self.has_lm_head:
            hidden = self.lm_head(hidden)
        out.last_hidden_state = hidden
        return out

    @staticmethod
    def _sequence_length(hidden_states: torch.Tensor) -> int:
        if hidden_states.dim() >= 2:
            return int(hidden_states.shape[1])
        return 1

    def _build_cache_position(
        self,
        seq_len: int,
        last_cache_position: Optional[int],
        expected_start: int,
    ) -> torch.Tensor:
        if last_cache_position is None:
            start = expected_start
        else:
            start = int(last_cache_position) - seq_len + 1
            if start < 0:
                start = 0

        return torch.arange(
            start,
            start + seq_len,
            device=self.device,
            dtype=torch.long,
        )

    def _store_cache_entry(
        self,
        request_id: str,
        past_key_values: Any,
        tokens_seen: int,
    ) -> None:
        if past_key_values is None:
            raise RuntimeError(
                f"Model did not return past_key_values for request {request_id}"
            )
        with self._cache_lock:
            if request_id in self._kv_cache_by_request:
                self._kv_cache_by_request.move_to_end(request_id)
            elif len(self._kv_cache_by_request) >= self.max_cached_requests:
                evicted_id, _ = self._kv_cache_by_request.popitem(last=False)
                log.warning(f"Evicted KV cache for request {evicted_id} (LRU)")

            self._kv_cache_by_request[request_id] = CacheEntry(
                past_key_values=past_key_values,
                tokens_seen=tokens_seen,
                last_access_ts=time.time(),
            )

    @staticmethod
    def _renumber_layer_indices(layers: nn.ModuleList) -> None:
        """Ensure sliced decoder layers use contiguous local cache indices."""
        for idx, layer in enumerate(layers):
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = idx
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "layer_idx"):
                self_attn.layer_idx = idx

    def clear_request_cache(self, request_id: str) -> None:
        """Clear KV cache for a specific request."""
        with self._cache_lock:
            self._kv_cache_by_request.pop(request_id, None)

    def clear_all_cache(self) -> None:
        """Clear KV cache for all requests."""
        with self._cache_lock:
            self._kv_cache_by_request.clear()

    def unload(self) -> None:
        """Unload the current model shard and free memory."""
        self.clear_all_cache()
        self.model = None
        self.lm_head = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("Shard unloaded, memory freed")

    def get_layer_info(self) -> dict:
        """Get information about the currently loaded shard."""
        with self._cache_lock:
            cached_requests = len(self._kv_cache_by_request)
        return {
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "num_layers": self.end_layer - self.start_layer,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "loaded": self._loaded,
            "device": str(self.device),
            "cached_requests": cached_requests,
        }
