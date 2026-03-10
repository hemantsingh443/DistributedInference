"""Model adapters for extracting standardized architecture specs.

HuggingFace models have widely varying configuration formats (e.g.,
Llama vs Qwen vs Phi). These adapters normalize the AutoConfig into a
unified ModelSpec dataclass used for VRAM calculations and weight slicing.
"""

from dataclasses import dataclass
from typing import Optional

from transformers import AutoConfig

from distributed_inference.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelSpec:
    """Standardized neural network geometry for a loaded model."""
    vocab_size: int
    hidden_size: int
    num_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_prefix: str
    embed_prefix: str
    final_norm_prefix: str
    lm_head_prefix: str
    tie_word_embeddings: bool


def get_model_spec(model_name: str) -> ModelSpec:
    """Create a standardized ModelSpec from a HuggingFace model config.

    Dynamically reads the AutoConfig and maps the architecture's specific
    weight key prefixes and hyperparameter geometries.

    Args:
        model_name: HuggingFace hub model ID or local path.

    Returns:
        Populated ModelSpec object.

    Raises:
        ValueError: If the architecture is fundamentally unsupported.
    """
    log.info(f"Extracting generic ModelSpec for architecture: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "unknown")

    # Unified base geometry extraction with fallbacks
    vocab_size = getattr(config, "vocab_size", 32000)
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 4096))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
    num_attention_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 0))
    # KV heads might be less than attention heads in GQA architectures
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

    if num_layers == 0 or num_attention_heads == 0:
        raise ValueError(f"Could not extract layer/head counts from config for {model_name}")

    spec = ModelSpec(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        layer_prefix="",
        embed_prefix="",
        final_norm_prefix="",
        lm_head_prefix="",
        tie_word_embeddings=tie_word_embeddings,
    )

    # Route weight key prefixes by architecture
    if model_type in ("llama", "mistral", "qwen2", "gemma", "gemma2"):
        spec.layer_prefix = "model.layers."
        spec.embed_prefix = "model.embed_tokens."
        spec.final_norm_prefix = "model.norm."
        spec.lm_head_prefix = "lm_head."
    elif model_type in ("phi", "gpt2"):
        # Phi and GPT2 usually use transformer.h. prefix
        spec.layer_prefix = "transformer.h."
        spec.embed_prefix = "transformer.embd.wte."
        
        # Phi-3 could use different norms but typically fallback below
        if hasattr(config, "ln_f") or model_type == "gpt2":
            spec.final_norm_prefix = "transformer.ln_f."
        else:
            spec.final_norm_prefix = "model.norm." # Backup
            
        spec.lm_head_prefix = "lm_head."
    else:
        # Fallback to Llama style as default for unknown Decoder models
        log.warning(f"Unknown model_type '{model_type}', falling back to Llama-style prefixes")
        spec.layer_prefix = "model.layers."
        spec.embed_prefix = "model.embed_tokens."
        spec.final_norm_prefix = "model.norm."
        spec.lm_head_prefix = "lm_head."

    log.debug(
        f"ModelSpec '{model_type}': layers={spec.num_layers}, "
        f"vocab={spec.vocab_size}, hidden={spec.hidden_size}, "
        f"q_heads={spec.num_attention_heads}, kv_heads={spec.num_key_value_heads}"
    )
    return spec
