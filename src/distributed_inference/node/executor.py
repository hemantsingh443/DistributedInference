"""Model shard executor for running forward passes on assigned layers.

Loads specific transformer layers from a HuggingFace model and runs
inference on them. Handles both first-node (embedding) and last-node
(lm_head) special components.

Designed for transformers 5.x which requires using the model's own
forward() method rather than calling individual layers directly.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from distributed_inference.common.logging import get_logger
from distributed_inference.node.resources import get_device, get_vram_usage_mb

log = get_logger(__name__)


class ShardExecutor:
    """Executes forward passes on a subset of model layers.

    Loads specific transformer layers from a pre-trained model and runs
    input tensors through them. Uses the model's own forward() method
    to ensure compatibility with transformers 5.x internal masking
    and positional embedding logic.
    """

    def __init__(self, device_type: str = "cpu"):
        self.device = get_device(device_type)
        self.model = None            # LlamaModel (the inner model)
        self.lm_head = None          # Linear projection to vocab
        self.model_config = None

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

        Uses the full model's internal forward() for compatibility with
        transformers 5.x. Loads the complete model, slices layers, and
        updates config.num_hidden_layers to match.

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

        # Load the full model
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # Slice layers to only the ones we need
        num_layers = end_layer - start_layer
        full_model.model.layers = nn.ModuleList(
            list(full_model.model.layers[start_layer:end_layer])
        )
        full_model.config.num_hidden_layers = num_layers
        full_model.config.use_cache = False

        # Store the inner model (LlamaModel) and optionally lm_head
        self.model = full_model.model

        if has_lm_head:
            self.lm_head = full_model.lm_head
        else:
            # Intermediate nodes: skip final RMSNorm (only last node applies it)
            self.model.norm = nn.Identity()
            self.lm_head = None

        # Move to target device
        self.model = self.model.to(self.device)
        if self.lm_head is not None:
            self.lm_head = self.lm_head.to(self.device)

        # Free the wrapper
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = True
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

        # Use the model's own forward() for proper internal handling
        # of causal masks, rotary embeddings, etc.
        if self.has_embedding:
            # First node: input is token IDs
            out = self.model(
                input_ids=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        else:
            # Intermediate/last node: input is hidden states
            out = self.model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        hidden_states = out.last_hidden_state

        # If we have the lm_head, project to vocabulary logits
        if self.has_lm_head:
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

    def unload(self) -> None:
        """Unload the current model shard and free memory."""
        self.model = None
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
