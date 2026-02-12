"""Model shard executor for running forward passes on assigned layers.

Loads specific transformer layers from a HuggingFace model and runs
inference on them. Handles both first-node (embedding) and last-node
(lm_head) special components.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from distributed_inference.common.logging import get_logger
from distributed_inference.node.resources import get_device, get_vram_usage_mb

log = get_logger(__name__)


class ShardExecutor:
    """Executes forward passes on a subset of model layers.

    Loads specific transformer layers from a pre-trained model and runs
    input tensors through them. Supports holding the embedding layer
    (for the first node) and lm_head (for the last node).
    """

    def __init__(self, device_type: str = "cpu"):
        self.device = get_device(device_type)
        self.layers: Optional[nn.ModuleList] = None
        self.embed_tokens: Optional[nn.Embedding] = None
        self.input_layernorm: Optional[nn.Module] = None  # model.norm (final)
        self.lm_head: Optional[nn.Linear] = None
        self.model_config: Optional[AutoConfig] = None

        self.start_layer: int = 0
        self.end_layer: int = 0
        self.has_embedding: bool = False
        self.has_lm_head: bool = False
        self._loaded: bool = False

    @property
    def loaded(self) -> bool:
        return self._loaded

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

        # Load model config
        self.model_config = AutoConfig.from_pretrained(model_name)

        # Load the full model to CPU first, then extract needed parts
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # Extract the layers we need
        self.layers = nn.ModuleList(
            full_model.model.layers[start_layer:end_layer]
        )

        if has_embedding:
            self.embed_tokens = full_model.model.embed_tokens

        if has_lm_head:
            self.input_layernorm = full_model.model.norm  # Final RMSNorm
            self.lm_head = full_model.lm_head

        # Move extracted components to device
        self.layers = self.layers.to(self.device)
        if self.embed_tokens is not None:
            self.embed_tokens = self.embed_tokens.to(self.device)
        if self.input_layernorm is not None:
            self.input_layernorm = self.input_layernorm.to(self.device)
        if self.lm_head is not None:
            self.lm_head = self.lm_head.to(self.device)

        # Delete the full model to free memory
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = True
        load_time_ms = (time.time() - start_time) * 1000
        vram_used = get_vram_usage_mb()

        num_layers = end_layer - start_layer
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
    ) -> torch.Tensor:
        """Run forward pass through the loaded layers.

        Args:
            hidden_states: Input tensor. If this node has the embedding layer,
                this should be input_ids (LongTensor). Otherwise, it's the
                hidden states from the previous node.
            attention_mask: Attention mask tensor.
            position_ids: Position IDs tensor.

        Returns:
            Output hidden states (or logits if this node has lm_head).
        """
        if not self._loaded:
            raise RuntimeError("No shard loaded. Call load_shard() first.")

        # Move inputs to device
        hidden_states = hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        # If we have the embedding layer, convert input_ids to embeddings
        if self.has_embedding:
            hidden_states = self.embed_tokens(hidden_states)

        # Prepare causal attention mask for the transformer layers
        # The HF LlamaModel internally creates a 4D causal mask
        # We need to replicate this for our sliced layers
        batch_size, seq_len = hidden_states.shape[:2]

        if attention_mask is not None:
            # Create a 4D causal attention mask
            # Shape: (batch_size, 1, seq_len, seq_len)
            causal_mask = self._make_causal_mask(
                batch_size, seq_len, hidden_states.dtype, hidden_states.device,
                attention_mask,
            )
        else:
            causal_mask = None

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Run through transformer layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        # If we have the lm_head, apply final norm + projection
        if self.has_lm_head:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

    def _make_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create a 4D causal attention mask.

        Follows the same pattern as HuggingFace's LlamaModel to ensure
        compatibility with the transformer layers.
        """
        # Create causal mask: upper triangular with -inf
        min_val = torch.finfo(dtype).min
        causal_mask = torch.full(
            (seq_len, seq_len), min_val, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # Expand to (batch_size, 1, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        if attention_mask is not None:
            # Combine with padding mask
            # attention_mask shape: (batch_size, seq_len), 1=attend, 0=mask
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_len, seq_len
            )
            inverted = (1.0 - expanded_mask.to(dtype)) * min_val
            causal_mask = causal_mask + inverted

        return causal_mask

    def unload(self) -> None:
        """Unload the current model shard and free memory."""
        self.layers = None
        self.embed_tokens = None
        self.input_layernorm = None
        self.lm_head = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("Shard unloaded, memory freed")

    def get_layer_info(self) -> dict:
        """Get information about the currently loaded shard."""
        return {
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "num_layers": self.end_layer - self.start_layer,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "loaded": self._loaded,
            "device": str(self.device),
        }
