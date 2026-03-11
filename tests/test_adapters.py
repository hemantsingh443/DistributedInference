import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from distributed_inference.coordinator.adapters import get_model_spec

models_to_test = [
    "Qwen/Qwen1.5-0.5B",
    "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-2b"
]

for model in models_to_test:
    print(f"\n--- Testing ModelSpec for {model} ---")
    try:
        spec = get_model_spec(model)
        print(f"  Vocab Size: {spec.vocab_size}")
        print(f"  Hidden Size: {spec.hidden_size}")
        print(f"  Layers: {spec.num_layers}")
        print(f"  Attention Heads: {spec.num_attention_heads}")
        print(f"  KV Heads: {spec.num_key_value_heads}")
        print(f"  Layer Prefix: {spec.layer_prefix}")
        print(f"  Embed Prefix: {spec.embed_prefix}")
        print(f"  Final Norm Prefix: {spec.final_norm_prefix}")
        print(f"  LM Head Prefix: {spec.lm_head_prefix}")
        print(f"  Dtype Bytes: {spec.dtype_bytes}")
    except Exception as e:
        print(f"  FAILED: {e}")
        
print("\nAll model specs tested!")
