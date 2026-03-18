#!/usr/bin/env python3
"""
Export EmbeddingGemma-300M float32 weights to .bin files for the C engine.

Downloads the sentence-transformers model from HuggingFace and exports:
  - Transformer weights (24 layers, Gemma 3 architecture)
  - Two dense projection layers (768→3072→768)
  - Tokenizer (SentencePiece, 262K vocab)

Architecture:
  Gemma3TextModel (bidirectional) → mean pool → Dense(768→3072) → Dense(3072→768) → L2 norm

Output layout:
  models/embeddinggemma/embedding.bin   — transformer + projection weights
  models/embeddinggemma/tokenizer.bin   — SentencePiece tokenizer

Usage:
    python scripts/export-weights.py
    python scripts/export-weights.py --output-dir /tmp/emb

Requirements:
    pip install -r scripts/requirements.txt
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models"))

MAGIC = b"MWTS"
VERSION = 5

HF_MODEL = "google/embeddinggemma-300m"


def write_bin(weights, config, output_path):
    """Write float32 weights + config to binary file."""
    tensors_meta = []
    tensors_data = []
    current_offset = 0

    # Embed config as _config tensor
    config_json = json.dumps(config, separators=(",", ":")).encode("utf-8")
    tensors_meta.append({
        "name": "_config",
        "shape": [len(config_json)],
        "dtype": "uint8",
        "offset": current_offset,
        "size": len(config_json),
    })
    tensors_data.append(config_json)
    current_offset += len(config_json)

    for name, arr in weights.items():
        arr = arr.astype(np.float32)
        raw = arr.tobytes()
        tensors_meta.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32",
            "offset": current_offset,
            "size": len(raw),
        })
        tensors_data.append(raw)
        current_offset += len(raw)

    header_json = json.dumps(tensors_meta, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for data in tensors_data:
            f.write(data)

    weight_tensors = [t for t in tensors_meta if t["name"] != "_config"]
    total_params = sum(int(np.prod(t["shape"])) for t in weight_tensors)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {output_path}: {len(weight_tensors)} tensors, {total_params:,} params, {size_mb:.1f} MB")


def export_tokenizer(output_dir):
    """Download and convert the EmbeddingGemma tokenizer."""
    tokenizer_bin = os.path.join(output_dir, "tokenizer.bin")
    if os.path.exists(tokenizer_bin):
        print(f"  tokenizer.bin already exists, skipping")
        return

    from transformers import AutoTokenizer

    print(f"  Downloading tokenizer from {HF_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)

    # Save tokenizer.json then convert to our binary format
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        tokenizer_json = os.path.join(tmp, "tokenizer.json")

        sys.path.insert(0, SCRIPT_DIR)
        from convert_tokenizer import convert_huggingface_json
        convert_huggingface_json(tokenizer_json, tokenizer_bin)


def main():
    parser = argparse.ArgumentParser(
        description="Export EmbeddingGemma-300M float32 weights for the C engine"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(MODELS_DIR, "embeddinggemma"),
        help="Output directory (default: models/embeddinggemma)",
    )
    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    print(f"Loading {HF_MODEL}...")
    st_model = SentenceTransformer(HF_MODEL, trust_remote_code=True, device="cpu")

    # Module 0: Transformer (Gemma3TextModel)
    transformer = st_model[0]
    hf_config = transformer.auto_model.config
    sd = transformer.state_dict()

    # Module 2: Dense(768 → 3072), Module 3: Dense(3072 → 768)
    dense1_sd = st_model[2].state_dict()
    dense2_sd = st_model[3].state_dict()

    num_layers = hf_config.num_hidden_layers
    hidden_size = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.head_dim
    intermediate_size = hf_config.intermediate_size
    vocab_size = hf_config.vocab_size
    max_seq_len = hf_config.max_position_embeddings
    rms_norm_eps = hf_config.rms_norm_eps
    rope_theta_sliding = 10000.0
    rope_theta_full = 1000000.0

    # Sliding window config — every 6th layer (index 5,11,17,23) is full attention
    sliding_window = hf_config.sliding_window

    config = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta_sliding": rope_theta_sliding,
        "rope_theta_full": rope_theta_full,
        "sliding_window": sliding_window,
        "dense1_out": 3072,
        "dense2_out": 768,
    }

    print(f"  Architecture: hidden={hidden_size} heads={num_heads} kv_heads={num_kv_heads} "
          f"hd={head_dim} mlp={intermediate_size} layers={num_layers} "
          f"vocab={vocab_size} max_seq={max_seq_len}")

    # --- Extract weights ---
    weights = {}

    # Token embeddings
    weights["embed_tokens.weight"] = sd["auto_model.embed_tokens.weight"].numpy()

    # Final RMSNorm
    weights["norm.weight"] = sd["auto_model.norm.weight"].numpy()

    # Transformer layers
    for i in range(num_layers):
        pfx_src = f"auto_model.layers.{i}"
        pfx_dst = f"layers.{i}"

        # Attention projections (transpose to [in, out] for our linear: y = x @ W)
        weights[f"{pfx_dst}.self_attn.q_proj.weight"] = (
            sd[f"{pfx_src}.self_attn.q_proj.weight"].numpy().T
        )
        weights[f"{pfx_dst}.self_attn.k_proj.weight"] = (
            sd[f"{pfx_src}.self_attn.k_proj.weight"].numpy().T
        )
        weights[f"{pfx_dst}.self_attn.v_proj.weight"] = (
            sd[f"{pfx_src}.self_attn.v_proj.weight"].numpy().T
        )
        weights[f"{pfx_dst}.self_attn.o_proj.weight"] = (
            sd[f"{pfx_src}.self_attn.o_proj.weight"].numpy().T
        )

        # QK norms
        weights[f"{pfx_dst}.self_attn.q_norm.weight"] = (
            sd[f"{pfx_src}.self_attn.q_norm.weight"].numpy()
        )
        weights[f"{pfx_dst}.self_attn.k_norm.weight"] = (
            sd[f"{pfx_src}.self_attn.k_norm.weight"].numpy()
        )

        # Sandwich norms (4 per layer)
        weights[f"{pfx_dst}.input_layernorm.weight"] = (
            sd[f"{pfx_src}.input_layernorm.weight"].numpy()
        )
        weights[f"{pfx_dst}.post_attention_layernorm.weight"] = (
            sd[f"{pfx_src}.post_attention_layernorm.weight"].numpy()
        )
        weights[f"{pfx_dst}.pre_feedforward_layernorm.weight"] = (
            sd[f"{pfx_src}.pre_feedforward_layernorm.weight"].numpy()
        )
        weights[f"{pfx_dst}.post_feedforward_layernorm.weight"] = (
            sd[f"{pfx_src}.post_feedforward_layernorm.weight"].numpy()
        )

        # MLP (transpose for our linear convention)
        weights[f"{pfx_dst}.mlp.gate_proj.weight"] = (
            sd[f"{pfx_src}.mlp.gate_proj.weight"].numpy().T
        )
        weights[f"{pfx_dst}.mlp.up_proj.weight"] = (
            sd[f"{pfx_src}.mlp.up_proj.weight"].numpy().T
        )
        weights[f"{pfx_dst}.mlp.down_proj.weight"] = (
            sd[f"{pfx_src}.mlp.down_proj.weight"].numpy().T
        )

    # Dense projection layers (sentence-transformers head)
    weights["dense1.weight"] = dense1_sd["linear.weight"].numpy().T
    weights["dense2.weight"] = dense2_sd["linear.weight"].numpy().T

    print("Exporting embedding model...")
    write_bin(weights, config, os.path.join(output_dir, "embedding.bin"))

    print("Exporting tokenizer...")
    export_tokenizer(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
