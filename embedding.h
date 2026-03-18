/*
 * embedding.h — EmbeddingGemma-300M sentence embedding + intent recognition.
 *
 * Zero-dependency C implementation of google/embeddinggemma-300m
 * (Gemma 3 bidirectional transformer + mean pooling + dense projection).
 *
 * Produces 768-dimensional L2-normalized sentence embeddings suitable
 * for cosine-similarity-based intent matching.
 *
 * Concurrency model:
 *   embedding_model is immutable after creation — share it freely across
 *   threads with no synchronization.  Each concurrent request gets its own
 *   embedding_state which holds mutable scratch buffers.
 *
 *   intent_recognizer is immutable during processing — register all intents
 *   before calling intent_recognizer_process from multiple threads.  Each
 *   concurrent process call needs its own embedding_state.
 *
 * Usage:
 *   embedding_model *model = embedding_model_load("models/embeddinggemma");
 *   embedding_state *state = embedding_state_create(model);
 *
 *   float emb[768];
 *   embedding_model_embed(model, state, "hello world", emb, 768);
 *
 *   embedding_state_free(state);
 *   embedding_model_free(model);
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Embedding model (immutable, thread-safe, load once) ───────── */

typedef struct embedding_model embedding_model;

/* Load embedding model from directory containing embedding.bin + tokenizer.bin.
 * Returns NULL on failure. */
embedding_model *embedding_model_load(const char *model_dir);

/* Free model and all weight data. */
void embedding_model_free(embedding_model *model);

/* ── Embedding state (mutable, one per concurrent request) ─────── */

typedef struct embedding_state embedding_state;

/* Create inference state for the given model.
 * Lightweight — allocates scratch buffers for the transformer forward pass.
 * Create one per thread or per request; do not share across concurrent calls. */
embedding_state *embedding_state_create(const embedding_model *model);

/* Free state and scratch buffers. */
void embedding_state_free(embedding_state *state);

/* ── Embedding ─────────────────────────────────────────────────── */

/* Embed text → 768-dim L2-normalized vector.
 * model:  shared, read-only — safe to use from multiple threads.
 * state:  per-request scratch — must not be shared across concurrent calls.
 * `out` must point to at least `out_dim` floats (768 for full embeddings).
 * Returns 0 on success, -1 on error. */
int embedding_model_embed(const embedding_model *model,
                          embedding_state *state,
                          const char *text,
                          float *out, int out_dim);

/* Cosine similarity between two L2-normalized embeddings. */
float embedding_cosine_similarity(const float *a, const float *b, int dim);

/* ── Intent recognizer ─────────────────────────────────────────── */

typedef struct intent_recognizer intent_recognizer;

typedef void (*intent_callback)(const char *trigger_phrase,
                                const char *utterance,
                                float similarity,
                                void *user_data);

/* Create intent recognizer backed by an embedding model.
 * Requires an embedding_state for computing trigger phrase embeddings
 * during registration. */
intent_recognizer *intent_recognizer_create(const embedding_model *model,
                                            embedding_state *state,
                                            float threshold);

/* Register an intent trigger phrase.  The embedding is computed immediately
 * using the state provided at creation.  Not thread-safe — register all
 * intents before calling intent_recognizer_process from multiple threads. */
int intent_recognizer_register(intent_recognizer *ir,
                               const char *trigger_phrase,
                               intent_callback callback,
                               void *user_data);

/* Remove a registered intent. Returns 0 if found and removed.
 * Not thread-safe — do not call concurrently with process. */
int intent_recognizer_unregister(intent_recognizer *ir,
                                 const char *trigger_phrase);

/* Match utterance against registered intents.
 * Fires callback for best match if similarity >= threshold.
 * model:  shared, read-only.
 * state:  per-request scratch — must not be shared across concurrent calls.
 * Returns 1 if an intent matched, 0 otherwise.
 * Thread-safe when intents are not being modified. */
int intent_recognizer_process(const intent_recognizer *ir,
                              const embedding_model *model,
                              embedding_state *state,
                              const char *utterance);

void intent_recognizer_set_threshold(intent_recognizer *ir, float threshold);
float intent_recognizer_get_threshold(const intent_recognizer *ir);
void intent_recognizer_free(intent_recognizer *ir);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDDING_H */
