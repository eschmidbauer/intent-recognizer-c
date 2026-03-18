# intent-recognizer-c

A zero-dependency C implementation of semantic intent recognition using [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) sentence embeddings.

Registers trigger phrases, embeds them with a Gemma 3 transformer, and matches utterances via cosine similarity. All weights are **float32**.

## Getting Started

### 1. Install Python dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Export model weights

```bash
python scripts/export-weights.py
```

Output goes to `models/embeddinggemma/` containing:

- `embedding.bin` — float32 transformer + projection weights (~1.2 GB)
- `tokenizer.bin` — SentencePiece tokenizer (262K vocab)

### 3. Build and test

```bash
make
./test_embedding models/embeddinggemma
```

## C API

```c
#include "embedding.h"

// Load model (immutable, thread-safe, load once)
embedding_model *model = embedding_model_load("models/embeddinggemma");

// Create per-thread state (mutable scratch buffers)
// Second arg caps sequence length: 128 ≈ 5.7 MB, 0 = model max (2048 ≈ 92 MB)
embedding_state *state = embedding_state_create(model, 128);

// Get a 768-dim L2-normalized embedding
float emb[768];
embedding_model_embed(model, state, "turn on the lights", emb, 768);

// Intent recognition
intent_recognizer *ir = intent_recognizer_create(model, state, 0.7f);
intent_recognizer_register(ir, "turn on the lights", my_callback, NULL);
intent_recognizer_register(ir, "what is the weather", my_callback, NULL);

// Process from any thread (with its own state)
intent_recognizer_process(ir, model, state, "switch on the lights");

intent_recognizer_free(ir);
embedding_state_free(state);
embedding_model_free(model);
```

## Concurrency Model

- **`embedding_model`** — immutable after load, share across threads
- **`embedding_state`** — mutable scratch buffers, one per concurrent call. Pass `max_seq` to control memory usage (~5.7 MB at 128 vs ~92 MB at 2048). For intent recognition, 64-128 is typically sufficient
- **`intent_recognizer`** — register intents during setup (single-threaded), then `process` is thread-safe with separate states

## Architecture

EmbeddingGemma-300M is a Gemma 3 bidirectional transformer:

1. Tokenize text (SentencePiece, 262K vocab)
2. Embed tokens + scale by sqrt(768)
3. 24 transformer layers (GQA 3Q/1KV, head_dim=256, RMSNorm, gated GELU MLP, RoPE)
4. Mean pool across sequence
5. Dense 768 → 3072 → 768
6. L2 normalize
