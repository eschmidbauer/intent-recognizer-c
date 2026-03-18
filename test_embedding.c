/*
 * test_embedding.c — Test EmbeddingGemma-300M embedding + intent recognition.
 *
 * Usage: ./test_embedding <model_dir>
 * Example: ./test_embedding models/embeddinggemma
 */

#include "embedding.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EMB_DIM 768

static void print_embedding(const char *label, const float *emb, int n) {
    printf("  %s first %d: [", label, n);
    for (int i = 0; i < n; i++) printf("%s%.6f", i ? ", " : "", emb[i]);
    printf("]\n");
}

static float vec_norm(const float *v, int dim) {
    float s = 0;
    for (int i = 0; i < dim; i++) s += v[i] * v[i];
    return sqrtf(s);
}

/* Reference values from Python (sentence-transformers) */
static const float ref_first8_lights[] = {
    -0.188852f, 0.019110f, 0.053933f, 0.006253f,
    -0.017149f, 0.037513f, -0.021670f, 0.015921f
};
static const float ref_sim_lights = 0.958773f;
static const float ref_sim_weather = 0.407931f;

static void on_intent(const char *trigger, const char *utterance,
                      float similarity, void *user_data) {
    (void)user_data;
    printf("  INTENT MATCH: trigger=\"%s\" utterance=\"%s\" sim=%.4f\n",
           trigger, utterance, similarity);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]);
        return 1;
    }

    printf("Loading embedding model from %s...\n", argv[1]);
    embedding_model *model = embedding_model_load(argv[1]);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    embedding_state *state = embedding_state_create(model);

    /* --- Test 1: Basic embedding --- */
    printf("\n=== Test 1: Embedding generation ===\n");

    float emb1[EMB_DIM], emb2[EMB_DIM], emb3[EMB_DIM];

    printf("Embedding \"turn on the lights\"...\n");
    if (embedding_model_embed(model, state, "turn on the lights", emb1, EMB_DIM) != 0) {
        fprintf(stderr, "Failed to embed\n"); return 1;
    }
    print_embedding("", emb1, 8);
    printf("  Norm: %.6f\n", vec_norm(emb1, EMB_DIM));

    printf("Embedding \"switch on the lights\"...\n");
    embedding_model_embed(model, state, "switch on the lights", emb2, EMB_DIM);
    print_embedding("", emb2, 8);

    printf("Embedding \"what is the weather\"...\n");
    embedding_model_embed(model, state, "what is the weather", emb3, EMB_DIM);
    print_embedding("", emb3, 8);

    /* --- Test 2: Similarity --- */
    printf("\n=== Test 2: Cosine similarity ===\n");
    float sim_12 = embedding_cosine_similarity(emb1, emb2, EMB_DIM);
    float sim_13 = embedding_cosine_similarity(emb1, emb3, EMB_DIM);
    printf("  sim(\"turn on the lights\", \"switch on the lights\") = %.6f (ref: %.6f)\n",
           sim_12, ref_sim_lights);
    printf("  sim(\"turn on the lights\", \"what is the weather\")  = %.6f (ref: %.6f)\n",
           sim_13, ref_sim_weather);

    /* Check against reference */
    printf("\n=== Test 3: Reference comparison ===\n");
    float max_diff = 0;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(emb1[i] - ref_first8_lights[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max diff in first 8 dims: %.6f\n", max_diff);
    printf("  Similarity diff (lights):  %.6f\n", fabsf(sim_12 - ref_sim_lights));
    printf("  Similarity diff (weather): %.6f\n", fabsf(sim_13 - ref_sim_weather));

    int pass = (max_diff < 0.05f) && (fabsf(sim_12 - ref_sim_lights) < 0.1f);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL (embeddings differ from reference)");

    /* --- Test 4: Intent recognition --- */
    printf("\n=== Test 4: Intent recognition ===\n");
    intent_recognizer *ir = intent_recognizer_create(model, state, 0.7f);

    printf("Registering intents...\n");
    intent_recognizer_register(ir, "turn on the lights", on_intent, NULL);
    intent_recognizer_register(ir, "what is the weather", on_intent, NULL);

    printf("Processing \"switch on the lights\":\n");
    int matched = intent_recognizer_process(ir, model, state, "switch on the lights");
    printf("  Matched: %s\n", matched ? "yes" : "no");

    printf("Processing \"play some music\":\n");
    matched = intent_recognizer_process(ir, model, state, "play some music");
    printf("  Matched: %s\n", matched ? "yes" : "no");

    intent_recognizer_free(ir);
    embedding_state_free(state);
    embedding_model_free(model);

    printf("\nDone.\n");
    return pass ? 0 : 1;
}
