/*
 * embedding.c — EmbeddingGemma-300M sentence embedding + intent recognition.
 *
 * Gemma 3 bidirectional transformer (24 layers, GQA 3Q/1KV, head_dim=256)
 * with mean pooling and dense projection to 768-dim L2-normalized embeddings.
 */

#include "embedding.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ───────────────── SIMD detection ───────────────── */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SIMD_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define SIMD_AVX2 1
#elif defined(__SSE2__)
#include <xmmintrin.h>
#include <emmintrin.h>
#define SIMD_SSE2 1
#endif

/* ───────────────── Constants ───────────────── */

#define EMB_BOS_TOKEN  2
#define EMB_EOS_TOKEN  1
#define EMB_MAX_SEQ    2048
#define EMB_DIM        768

/* ───────────────── Internal types ───────────────── */

typedef struct {
    int hidden_size, num_heads, num_kv_heads, head_dim;
    int intermediate_size, num_layers, vocab_size, max_seq_len;
    float rms_norm_eps;
    float rope_theta_sliding, rope_theta_full;
    int sliding_window;
    int dense1_out, dense2_out;
} EmbConfig;

typedef struct {
    char *name;
    int shape[8], ndim, offset, size;
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    const uint8_t *data;
    uint8_t *file_buf;
    size_t file_size;
} WeightFile;

/* Tokenizer: maps token_id → byte string, and text → token_ids */
typedef struct {
    uint8_t **token_bytes;
    int *token_lens;
    int num_tokens;
} EmbTokenizer;

/* Hash map for fast text → token_id lookup during encoding */
#define HASH_BUCKETS (1 << 19) /* 524288 */

typedef struct HashEntry {
    const uint8_t *key;
    int key_len;
    int token_id;
    struct HashEntry *next;
} HashEntry;

typedef struct {
    HashEntry *buckets[HASH_BUCKETS];
    HashEntry *pool;
    int pool_size;
} HashMap;

/* Per-layer weights */
typedef struct {
    const float *input_ln_w, *post_attn_ln_w;
    const float *pre_ff_ln_w, *post_ff_ln_w;
    const float *q_proj, *k_proj, *v_proj, *o_proj;
    const float *q_norm, *k_norm;
    const float *gate_proj, *up_proj, *down_proj;
} EmbLayer;

struct embedding_model {
    EmbConfig cfg;
    const float *embed_w, *norm_w;
    const float *dense1_w, *dense2_w;
    EmbLayer *layers;
    EmbTokenizer *tok;
    HashMap *tok_map;
    WeightFile *wf;
};

/* State: mutable per-request scratch, one per concurrent call. */
struct embedding_state {
    /* Transformer scratch buffers (allocated for max_seq) */
    float *x;       /* [max_seq * hidden_size] hidden states */
    float *tmp;     /* [max_seq * hidden_size] temp for norms */
    float *attn;    /* [max_seq * hidden_size] attention output */
    float *mlp_buf; /* [max_seq * hidden_size] mlp output */
    /* Attention scratch */
    float *q;       /* [max_seq * num_heads * head_dim] */
    float *k;       /* [max_seq * num_kv_heads * head_dim] */
    float *v;       /* [max_seq * num_kv_heads * head_dim] */
    float *attn_out;/* [max_seq * num_heads * head_dim] */
    float *scores;  /* [max_seq] */
    /* MLP scratch */
    float *gate;    /* [max_seq * intermediate_size] */
    float *up;      /* [max_seq * intermediate_size] */
    /* Projection scratch */
    float *pooled;  /* [hidden_size] */
    float *d1;      /* [dense1_out] */
    float *d2;      /* [dense2_out] */
    /* Tokenizer scratch */
    int *tokens;    /* [max_seq] */
};

/* Intent recognizer */
typedef struct {
    char *trigger;
    float *embedding;
    intent_callback callback;
    void *user_data;
} Intent;

struct intent_recognizer {
    const embedding_model *model;
    embedding_state *reg_state;  /* state used during registration only */
    Intent *intents;
    int num_intents, cap_intents;
    float threshold;
};

/* ───────────────── Weight file reader ───────────────── */

static const char *skip_ws(const char *p, const char *e) {
    while (p < e && (*p==' '||*p=='\n'||*p=='\r'||*p=='\t')) p++;
    return p;
}

static const char *parse_str(const char *p, const char *e, char **out) {
    if (p >= e || *p != '"') return NULL;
    p++; const char *s = p;
    while (p < e && *p != '"') p++;
    if (p >= e) return NULL;
    size_t len = (size_t)(p - s);
    *out = (char *)malloc(len + 1);
    memcpy(*out, s, len); (*out)[len] = '\0';
    return p + 1;
}

static const char *parse_int64(const char *p, const char *e, int64_t *out) {
    int64_t v = 0; int neg = 0;
    if (p < e && *p == '-') { neg = 1; p++; }
    while (p < e && *p >= '0' && *p <= '9') { v = v*10 + (*p-'0'); p++; }
    *out = neg ? -v : v;
    return p;
}

static const char *parse_float(const char *p, const char *e, double *out) {
    char buf[64]; int i = 0;
    while (p < e && i < 63 && ((*p>='0'&&*p<='9')||*p=='.'||*p=='-'||*p=='e'||*p=='E'||*p=='+'))
        buf[i++] = *p++;
    buf[i] = '\0';
    *out = atof(buf);
    return p;
}

static WeightFile *load_weights(const char *path) {
#ifdef _WIN32
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "embedding: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); size_t fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(fsz); fread(buf, 1, fsz, f); fclose(f);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "embedding: cannot open %s\n", path); return NULL; }
    struct stat st; fstat(fd, &st); size_t fsz = (size_t)st.st_size;
    uint8_t *buf = (uint8_t *)mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) { fprintf(stderr, "embedding: mmap failed %s\n", path); return NULL; }
#endif
    if (fsz < 12 || memcmp(buf, "MWTS", 4) != 0) {
        fprintf(stderr, "embedding: bad magic in %s\n", path); return NULL;
    }
    uint32_t header_size; memcpy(&header_size, buf + 8, 4);
    const char *json = (const char *)(buf + 12);
    const char *json_end = json + header_size;
    const char *p = skip_ws(json, json_end);
    if (*p != '[') return NULL; p++;
    int count = 0;
    { const char *q = p; int d = 0;
      while (q < json_end) { if (*q=='{' && d++==0) count++; if (*q=='}') d--; q++; } }
    TensorInfo *tensors = (TensorInfo *)calloc(count, sizeof(TensorInfo));
    int idx = 0;
    while (idx < count) {
        p = skip_ws(p, json_end); if (*p==',') p++; p = skip_ws(p, json_end);
        if (*p==']') break; if (*p!='{') break; p++;
        TensorInfo *t = &tensors[idx];
        while (p < json_end && *p != '}') {
            p = skip_ws(p, json_end); if (*p==',') { p++; p = skip_ws(p, json_end); }
            if (*p=='}') break;
            char *key = NULL; p = parse_str(p, json_end, &key);
            p = skip_ws(p, json_end); if (*p==':') p++; p = skip_ws(p, json_end);
            if (strcmp(key, "name") == 0) { p = parse_str(p, json_end, &t->name); }
            else if (strcmp(key, "shape") == 0) {
                p++; t->ndim = 0;
                while (p < json_end && *p != ']') {
                    p = skip_ws(p, json_end); if (*p==',') { p++; p = skip_ws(p, json_end); }
                    if (*p==']') break;
                    int64_t d; p = parse_int64(p, json_end, &d);
                    if (t->ndim < 8) t->shape[t->ndim++] = (int)d;
                }
                if (p < json_end) p++;
            } else if (strcmp(key, "dtype") == 0) { char *dt; p = parse_str(p, json_end, &dt); free(dt); }
            else if (strcmp(key, "offset") == 0) { int64_t v; p = parse_int64(p, json_end, &v); t->offset = (int)v; }
            else if (strcmp(key, "size") == 0) { int64_t v; p = parse_int64(p, json_end, &v); t->size = (int)v; }
            free(key);
        }
        if (p < json_end) p++; idx++;
    }
    WeightFile *wf = (WeightFile *)calloc(1, sizeof(WeightFile));
    wf->tensors = tensors; wf->num_tensors = idx;
    wf->data = buf + 12 + header_size;
    wf->file_buf = buf; wf->file_size = fsz;
    return wf;
}

static const TensorInfo *find_tensor(const WeightFile *wf, const char *name) {
    for (int i = 0; i < wf->num_tensors; i++)
        if (strcmp(wf->tensors[i].name, name) == 0) return &wf->tensors[i];
    return NULL;
}

static const float *get_weight(const WeightFile *wf, const char *name) {
    const TensorInfo *t = find_tensor(wf, name);
    if (!t) { fprintf(stderr, "embedding: weight not found: %s\n", name); exit(1); }
    return (const float *)(wf->data + t->offset);
}

static int load_emb_config(const WeightFile *wf, EmbConfig *cfg) {
    const TensorInfo *t = find_tensor(wf, "_config");
    if (!t) return -1;
    const char *js = (const char *)(wf->data + t->offset);
    const char *end = js + t->size;
    const char *p = skip_ws(js, end);
    if (*p != '{') return -1; p++;
    memset(cfg, 0, sizeof(*cfg));
    while (p < end && *p != '}') {
        p = skip_ws(p, end); if (*p==',') { p++; p = skip_ws(p, end); }
        if (*p=='}') break;
        char *key = NULL; p = parse_str(p, end, &key);
        p = skip_ws(p, end); if (*p==':') p++; p = skip_ws(p, end);
        /* float fields */
        if (strcmp(key, "rms_norm_eps") == 0 || strcmp(key, "rope_theta_sliding") == 0 ||
            strcmp(key, "rope_theta_full") == 0) {
            double fv; p = parse_float(p, end, &fv);
            if (strcmp(key, "rms_norm_eps") == 0) cfg->rms_norm_eps = (float)fv;
            else if (strcmp(key, "rope_theta_sliding") == 0) cfg->rope_theta_sliding = (float)fv;
            else cfg->rope_theta_full = (float)fv;
        } else {
            int64_t val; p = parse_int64(p, end, &val);
            if      (strcmp(key,"hidden_size")==0)       cfg->hidden_size = (int)val;
            else if (strcmp(key,"num_heads")==0)          cfg->num_heads = (int)val;
            else if (strcmp(key,"num_kv_heads")==0)       cfg->num_kv_heads = (int)val;
            else if (strcmp(key,"head_dim")==0)           cfg->head_dim = (int)val;
            else if (strcmp(key,"intermediate_size")==0)  cfg->intermediate_size = (int)val;
            else if (strcmp(key,"num_layers")==0)         cfg->num_layers = (int)val;
            else if (strcmp(key,"vocab_size")==0)         cfg->vocab_size = (int)val;
            else if (strcmp(key,"max_seq_len")==0)        cfg->max_seq_len = (int)val;
            else if (strcmp(key,"sliding_window")==0)     cfg->sliding_window = (int)val;
            else if (strcmp(key,"dense1_out")==0)         cfg->dense1_out = (int)val;
            else if (strcmp(key,"dense2_out")==0)         cfg->dense2_out = (int)val;
        }
        free(key);
    }
    return 0;
}

static void free_weights(WeightFile *wf) {
    if (!wf) return;
    for (int i = 0; i < wf->num_tensors; i++) free(wf->tensors[i].name);
    free(wf->tensors);
#ifdef _WIN32
    free(wf->file_buf);
#else
    munmap(wf->file_buf, wf->file_size);
#endif
    free(wf);
}

/* ───────────────── Tokenizer ───────────────── */

static EmbTokenizer *load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "embedding: cannot open tokenizer %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(fsz); fread(data, 1, fsz, f); fclose(f);

    /* First pass: count tokens */
    int count = 0; long off = 0;
    while (off < fsz) {
        uint8_t b = data[off++];
        int len = (b < 128) ? b : ((off < fsz) ? (data[off++]*128 + b - 128) : 0);
        off += len; count++;
    }

    EmbTokenizer *tok = (EmbTokenizer *)calloc(1, sizeof(EmbTokenizer));
    tok->token_bytes = (uint8_t **)calloc(count, sizeof(uint8_t *));
    tok->token_lens = (int *)calloc(count, sizeof(int));
    tok->num_tokens = count;

    /* Second pass: load tokens */
    off = 0; int idx = 0;
    while (off < fsz && idx < count) {
        uint8_t b = data[off++];
        int len = (b < 128) ? b : ((off < fsz) ? (data[off++]*128 + b - 128) : 0);
        tok->token_bytes[idx] = (uint8_t *)malloc(len + 1);
        if (len > 0) memcpy(tok->token_bytes[idx], data + off, len);
        tok->token_bytes[idx][len] = '\0';
        tok->token_lens[idx] = len;
        off += len; idx++;
    }
    free(data);
    return tok;
}

static void free_tokenizer(EmbTokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->num_tokens; i++) free(tok->token_bytes[i]);
    free(tok->token_bytes); free(tok->token_lens); free(tok);
}

/* Hash map for encoding */
static uint32_t hash_bytes(const uint8_t *data, int len) {
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < len; i++) { h ^= data[i]; h *= 0x01000193u; }
    return h;
}

static HashMap *build_token_map(const EmbTokenizer *tok) {
    HashMap *m = (HashMap *)calloc(1, sizeof(HashMap));
    m->pool = (HashEntry *)calloc(tok->num_tokens, sizeof(HashEntry));
    m->pool_size = tok->num_tokens;
    for (int i = 0; i < tok->num_tokens; i++) {
        if (tok->token_lens[i] == 0) continue;
        HashEntry *e = &m->pool[i];
        e->key = tok->token_bytes[i];
        e->key_len = tok->token_lens[i];
        e->token_id = i;
        uint32_t idx = hash_bytes(e->key, e->key_len) & (HASH_BUCKETS - 1);
        e->next = m->buckets[idx];
        m->buckets[idx] = e;
    }
    return m;
}

static int lookup_token(const HashMap *m, const uint8_t *data, int len) {
    uint32_t idx = hash_bytes(data, len) & (HASH_BUCKETS - 1);
    for (HashEntry *e = m->buckets[idx]; e; e = e->next)
        if (e->key_len == len && memcmp(e->key, data, len) == 0) return e->token_id;
    return -1;
}

static void free_token_map(HashMap *m) {
    if (!m) return;
    free(m->pool); free(m);
}

/* Greedy longest-match tokenization.
 * SentencePiece convention: spaces become ▁ (U+2581 = 0xE2 0x96 0x81).
 * First word has no ▁ prefix; subsequent words get ▁ prepended. */
static int tokenize(const HashMap *map, const char *text,
                    int *tokens, int max_tokens) {
    int tlen = (int)strlen(text);
    int buf_len = tlen * 3 + 4;
    uint8_t *buf = (uint8_t *)malloc(buf_len);
    int blen = 0;
    /* No leading ▁ — spaces become ▁ */
    for (int i = 0; i < tlen; i++) {
        if (text[i] == ' ') {
            buf[blen++] = 0xE2; buf[blen++] = 0x96; buf[blen++] = 0x81;
        } else {
            buf[blen++] = (uint8_t)text[i];
        }
    }

    int n = 0;
    tokens[n++] = EMB_BOS_TOKEN;

    int pos = 0;
    while (pos < blen && n < max_tokens - 1) {
        int best_len = 0, best_id = -1;
        int max_try = blen - pos;
        if (max_try > 64) max_try = 64; /* cap token length */
        for (int l = max_try; l >= 1; l--) {
            int id = lookup_token(map, buf + pos, l);
            if (id >= 0) { best_len = l; best_id = id; break; }
        }
        if (best_id >= 0) {
            tokens[n++] = best_id;
            pos += best_len;
        } else {
            pos++; /* skip unknown byte */
        }
    }

    tokens[n++] = EMB_EOS_TOKEN;
    free(buf);
    return n;
}

/* ───────────────── SIMD primitives ───────────────── */

/* dot product: sum(a[i]*b[i]) for i in [0,n) */
static inline float vec_dot(const float *a, const float *b, int n) {
    float sum = 0;
    int i = 0;
#if SIMD_NEON
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),   vld1q_f32(b+i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4), vld1q_f32(b+i+4));
    }
    for (; i + 4 <= n; i += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i), vld1q_f32(b+i));
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#elif SIMD_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    /* horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    sum = _mm_cvtss_f32(lo);
#elif SIMD_SSE2
    __m128 acc0s = _mm_setzero_ps(), acc1s = _mm_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128 a0 = _mm_loadu_ps(a+i),   b0 = _mm_loadu_ps(b+i);
        __m128 a1 = _mm_loadu_ps(a+i+4), b1 = _mm_loadu_ps(b+i+4);
        acc0s = _mm_add_ps(acc0s, _mm_mul_ps(a0, b0));
        acc1s = _mm_add_ps(acc1s, _mm_mul_ps(a1, b1));
    }
    for (; i + 4 <= n; i += 4)
        acc0s = _mm_add_ps(acc0s, _mm_mul_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i)));
    acc0s = _mm_add_ps(acc0s, acc1s);
    /* horizontal sum: SSE2 path */
    __m128 shuf = _mm_shuffle_ps(acc0s, acc0s, _MM_SHUFFLE(2,3,0,1));
    acc0s = _mm_add_ps(acc0s, shuf);
    shuf = _mm_shuffle_ps(acc0s, acc0s, _MM_SHUFFLE(0,0,3,3));
    acc0s = _mm_add_ps(acc0s, shuf);
    sum = _mm_cvtss_f32(acc0s);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* saxpy: y[i] += a * x[i] for i in [0,n) */
static inline void vec_saxpy(float *y, float a, const float *x, int n) {
    int i = 0;
#if SIMD_NEON
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y+i,   vfmaq_f32(vld1q_f32(y+i),   va, vld1q_f32(x+i)));
        vst1q_f32(y+i+4, vfmaq_f32(vld1q_f32(y+i+4), va, vld1q_f32(x+i+4)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y+i, vfmaq_f32(vld1q_f32(y+i), va, vld1q_f32(x+i)));
#elif SIMD_AVX2
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y+i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x+i), _mm256_loadu_ps(y+i)));
#elif SIMD_SSE2
    __m128 va = _mm_set1_ps(a);
    for (; i + 4 <= n; i += 4) {
        __m128 vy = _mm_loadu_ps(y+i);
        vy = _mm_add_ps(vy, _mm_mul_ps(va, _mm_loadu_ps(x+i)));
        _mm_storeu_ps(y+i, vy);
    }
#endif
    for (; i < n; i++) y[i] += a * x[i];
}

/* vector add: x[i] += y[i] for i in [0,n) */
static inline void vec_add(float *x, const float *y, int n) {
    int i = 0;
#if SIMD_NEON
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(x+i,   vaddq_f32(vld1q_f32(x+i),   vld1q_f32(y+i)));
        vst1q_f32(x+i+4, vaddq_f32(vld1q_f32(x+i+4), vld1q_f32(y+i+4)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(x+i, vaddq_f32(vld1q_f32(x+i), vld1q_f32(y+i)));
#elif SIMD_AVX2
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x+i, _mm256_add_ps(_mm256_loadu_ps(x+i), _mm256_loadu_ps(y+i)));
#elif SIMD_SSE2
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(x+i, _mm_add_ps(_mm_loadu_ps(x+i), _mm_loadu_ps(y+i)));
#endif
    for (; i < n; i++) x[i] += y[i];
}

/* sum of squares: sum(x[i]^2) for i in [0,n) */
static inline float vec_sum_sq(const float *x, int n) {
    return vec_dot(x, x, n);
}

/* scale: x[i] *= a for i in [0,n) */
static inline void vec_scale(float *x, float a, int n) {
    int i = 0;
#if SIMD_NEON
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(x+i,   vmulq_f32(vld1q_f32(x+i),   va));
        vst1q_f32(x+i+4, vmulq_f32(vld1q_f32(x+i+4), va));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(x+i, vmulq_f32(vld1q_f32(x+i), va));
#elif SIMD_AVX2
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x+i, _mm256_mul_ps(_mm256_loadu_ps(x+i), va));
#elif SIMD_SSE2
    __m128 va = _mm_set1_ps(a);
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(x+i, _mm_mul_ps(_mm_loadu_ps(x+i), va));
#endif
    for (; i < n; i++) x[i] *= a;
}

/* ───────────────── Math operations ───────────────── */

#define TILE 32

static void matmul(float *out, const float *a, const float *b, int M, int K, int N) {
    memset(out, 0, (size_t)M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += TILE)
      for (int k0 = 0; k0 < K; k0 += TILE)
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int imax = i0+TILE < M ? i0+TILE : M;
            int kmax = k0+TILE < K ? k0+TILE : K;
            int jmax = j0+TILE < N ? j0+TILE : N;
            for (int i = i0; i < imax; i++) {
                const float *ar = a + i*K;
                float *cr = out + i*N + j0;
                for (int k = k0; k < kmax; k++) {
                    vec_saxpy(cr, ar[k], b + k*N + j0, jmax - j0);
                }
            }
        }
}

static void linear(float *out, const float *x, const float *w,
                   int seq, int in_dim, int out_dim) {
    matmul(out, x, w, seq, in_dim, out_dim);
}

/* Gemma3 RMSNorm: output = x / rms(x) * (1 + weight) */
static void rms_norm(float *out, const float *x, const float *w,
                     int seq, int dim, float eps) {
    for (int i = 0; i < seq; i++) {
        const float *xi = x + i*dim;
        float *oi = out + i*dim;
        float ss = vec_sum_sq(xi, dim);
        float inv = 1.0f / sqrtf(ss / dim + eps);
        int j = 0;
#if SIMD_NEON
        float32x4_t vinv = vdupq_n_f32(inv);
        float32x4_t vone = vdupq_n_f32(1.0f);
        for (; j + 4 <= dim; j += 4) {
            float32x4_t vx = vld1q_f32(xi + j);
            float32x4_t vw = vld1q_f32(w + j);
            vst1q_f32(oi + j, vmulq_f32(vmulq_f32(vx, vinv), vaddq_f32(vone, vw)));
        }
#elif SIMD_AVX2
        __m256 vinv = _mm256_set1_ps(inv);
        __m256 vone = _mm256_set1_ps(1.0f);
        for (; j + 8 <= dim; j += 8) {
            __m256 vx = _mm256_loadu_ps(xi + j);
            __m256 vw = _mm256_loadu_ps(w + j);
            _mm256_storeu_ps(oi + j, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), _mm256_add_ps(vone, vw)));
        }
#elif SIMD_SSE2
        __m128 vinv = _mm_set1_ps(inv);
        __m128 vone = _mm_set1_ps(1.0f);
        for (; j + 4 <= dim; j += 4) {
            __m128 vx = _mm_loadu_ps(xi + j);
            __m128 vw = _mm_loadu_ps(w + j);
            _mm_storeu_ps(oi + j, _mm_mul_ps(_mm_mul_ps(vx, vinv), _mm_add_ps(vone, vw)));
        }
#endif
        for (; j < dim; j++) oi[j] = xi[j] * inv * (1.0f + w[j]);
    }
}

static inline float gelu_tanh_f(float x) {
    float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ───────────────── RoPE ───────────────── */

static void apply_rope_emb(float *q, int q_heads, float *k, int kv_heads,
                           int seq, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int s = 0; s < seq; s++) {
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2*d) / (float)head_dim);
            float angle = (float)s * freq;
            float co = cosf(angle), si = sinf(angle);
            /* Apply to all Q heads */
            for (int h = 0; h < q_heads; h++) {
                float *qh = q + (s*q_heads + h)*head_dim;
                float q0 = qh[2*d], q1 = qh[2*d+1];
                qh[2*d]   = q0*co - q1*si;
                qh[2*d+1] = q0*si + q1*co;
            }
            /* Apply to all KV heads */
            for (int h = 0; h < kv_heads; h++) {
                float *kh = k + (s*kv_heads + h)*head_dim;
                float k0 = kh[2*d], k1 = kh[2*d+1];
                kh[2*d]   = k0*co - k1*si;
                kh[2*d+1] = k0*si + k1*co;
            }
        }
    }
}

/* ───────────────── GQA Bidirectional Attention ───────────────── */

static void gqa_attention(float *out, const float *x, const EmbLayer *ly,
                          int seq, const EmbConfig *cfg, int layer_idx,
                          embedding_state *st) {
    int D = cfg->hidden_size;
    int H = cfg->num_heads;
    int KVH = cfg->num_kv_heads;
    int HD = cfg->head_dim;
    int heads_per_group = H / KVH;
    float eps = cfg->rms_norm_eps;

    float *q = st->q;
    float *k = st->k;
    float *v = st->v;
    float *attn_out = st->attn_out;
    float *scores = st->scores;

    /* Project Q, K, V */
    linear(q, x, ly->q_proj, seq, D, H * HD);
    linear(k, x, ly->k_proj, seq, D, KVH * HD);
    linear(v, x, ly->v_proj, seq, D, KVH * HD);

    /* QK norms: apply per-head RMSNorm */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < H; h++)
            rms_norm(q + (s*H+h)*HD, q + (s*H+h)*HD, ly->q_norm, 1, HD, eps);
        for (int h = 0; h < KVH; h++)
            rms_norm(k + (s*KVH+h)*HD, k + (s*KVH+h)*HD, ly->k_norm, 1, HD, eps);
    }

    /* RoPE — select theta based on layer type */
    /* Layers 5,11,17,23 are full attention; rest are sliding window */
    int is_full = ((layer_idx + 1) % 6 == 0);
    float theta = is_full ? cfg->rope_theta_full : cfg->rope_theta_sliding;
    apply_rope_emb(q, H, k, KVH, seq, HD, theta);

    /* Attention: each Q head attends to its corresponding KV head group */
    float scale = 1.0f / sqrtf((float)HD);
    memset(attn_out, 0, (size_t)seq * H * HD * sizeof(float));

    for (int h = 0; h < H; h++) {
        int kv_h = h / heads_per_group;
        for (int i = 0; i < seq; i++) {
            const float *qi = q + (i*H + h)*HD;
            /* Compute scores against all K positions */
            for (int j = 0; j < seq; j++) {
                const float *kj = k + (j*KVH + kv_h)*HD;
                scores[j] = vec_dot(qi, kj, HD) * scale;
            }

            /* Sliding window mask for non-full-attention layers */
            if (!is_full) {
                int win = cfg->sliding_window;
                for (int j = 0; j < seq; j++) {
                    int dist = i - j; if (dist < 0) dist = -dist;
                    if (dist >= win) scores[j] = -1e9f;
                }
            }

            softmax(scores, seq);

            /* Weighted sum of V */
            float *oi = attn_out + (i*H + h)*HD;
            for (int j = 0; j < seq; j++)
                vec_saxpy(oi, scores[j], v + (j*KVH + kv_h)*HD, HD);
        }
    }

    /* Output projection: [seq, H*HD] → [seq, D] */
    linear(out, attn_out, ly->o_proj, seq, H*HD, D);
}

/* ───────────────── Gated MLP ───────────────── */

static void gated_mlp(float *out, const float *x, const EmbLayer *ly,
                      int seq, int D, int mlp_dim, embedding_state *st) {
    float *gate = st->gate;
    float *up   = st->up;

    linear(gate, x, ly->gate_proj, seq, D, mlp_dim);
    linear(up,   x, ly->up_proj,   seq, D, mlp_dim);

    /* GELU(gate) * up */
    for (int i = 0; i < seq * mlp_dim; i++)
        gate[i] = gelu_tanh_f(gate[i]) * up[i];

    /* Down projection */
    linear(out, gate, ly->down_proj, seq, mlp_dim, D);
}

/* ───────────────── Transformer forward pass ───────────────── */

/* Forward pass using pre-allocated state buffers.
 * Returns pointer to st->x which holds [seq, D] hidden states. */
static float *emb_forward(const embedding_model *model, embedding_state *st,
                           const int *tokens, int seq) {
    const EmbConfig *c = &model->cfg;
    int D = c->hidden_size;
    float eps = c->rms_norm_eps;

    /* Token embedding lookup + scaling by sqrt(hidden_size).
     * Gemma3TextScaledWordEmbedding applies this internally. */
    float *x = st->x;
    float emb_scale = sqrtf((float)D);
    for (int i = 0; i < seq; i++) {
        const float *emb = model->embed_w + tokens[i] * D;
        float *xi = x + i * D;
        memcpy(xi, emb, D * sizeof(float));
        vec_scale(xi, emb_scale, D);
    }

    float *tmp     = st->tmp;
    float *attn    = st->attn;
    float *mlp_buf = st->mlp_buf;

    for (int l = 0; l < c->num_layers; l++) {
        const EmbLayer *ly = &model->layers[l];

        /* --- Attention block (sandwich norm) --- */
        rms_norm(tmp, x, ly->input_ln_w, seq, D, eps);
        gqa_attention(attn, tmp, ly, seq, c, l, st);
        rms_norm(attn, attn, ly->post_attn_ln_w, seq, D, eps);
        vec_add(x, attn, seq * D);

        /* --- MLP block (sandwich norm) --- */
        rms_norm(tmp, x, ly->pre_ff_ln_w, seq, D, eps);
        gated_mlp(mlp_buf, tmp, ly, seq, D, c->intermediate_size, st);
        rms_norm(mlp_buf, mlp_buf, ly->post_ff_ln_w, seq, D, eps);
        vec_add(x, mlp_buf, seq * D);
    }

    /* Final RMSNorm */
    rms_norm(x, x, model->norm_w, seq, D, eps);

    return x; /* [seq, D] — points into st->x, valid until next call */
}

/* ───────────────── Mean pooling + projection + L2 norm ───────────────── */

static void mean_pool(float *out, const float *hidden, int seq, int dim) {
    memset(out, 0, dim * sizeof(float));
    for (int i = 0; i < seq; i++)
        vec_add(out, hidden + i*dim, dim);
    vec_scale(out, 1.0f / (float)seq, dim);
}

static void l2_normalize(float *v, int dim) {
    float norm = sqrtf(vec_sum_sq(v, dim));
    if (norm > 0.0f)
        vec_scale(v, 1.0f / norm, dim);
}

/* ───────────────── Public API: Embedding Model ───────────────── */

embedding_model *embedding_model_load(const char *model_dir) {
    char path[1024];

    snprintf(path, sizeof(path), "%s/embedding.bin", model_dir);
    WeightFile *wf = load_weights(path);
    if (!wf) return NULL;

    snprintf(path, sizeof(path), "%s/tokenizer.bin", model_dir);
    EmbTokenizer *tok = load_tokenizer(path);
    if (!tok) { free_weights(wf); return NULL; }

    embedding_model *m = (embedding_model *)calloc(1, sizeof(embedding_model));
    m->wf = wf;
    m->tok = tok;
    m->tok_map = build_token_map(tok);

    if (load_emb_config(wf, &m->cfg) != 0) {
        fprintf(stderr, "embedding: failed to load config\n");
        embedding_model_free(m); return NULL;
    }

    EmbConfig *c = &m->cfg;

    /* Global weights */
    m->embed_w  = get_weight(wf, "embed_tokens.weight");
    m->norm_w   = get_weight(wf, "norm.weight");
    m->dense1_w = get_weight(wf, "dense1.weight");
    m->dense2_w = get_weight(wf, "dense2.weight");

    /* Layer weights */
    m->layers = (EmbLayer *)calloc(c->num_layers, sizeof(EmbLayer));
    for (int i = 0; i < c->num_layers; i++) {
        EmbLayer *ly = &m->layers[i];
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "layers.%d", i);
        char name[128];

#define W(field, suffix) \
    snprintf(name, sizeof(name), "%s.%s", pfx, suffix); \
    ly->field = get_weight(wf, name);

        W(input_ln_w,    "input_layernorm.weight");
        W(post_attn_ln_w,"post_attention_layernorm.weight");
        W(pre_ff_ln_w,   "pre_feedforward_layernorm.weight");
        W(post_ff_ln_w,  "post_feedforward_layernorm.weight");
        W(q_proj,        "self_attn.q_proj.weight");
        W(k_proj,        "self_attn.k_proj.weight");
        W(v_proj,        "self_attn.v_proj.weight");
        W(o_proj,        "self_attn.o_proj.weight");
        W(q_norm,        "self_attn.q_norm.weight");
        W(k_norm,        "self_attn.k_norm.weight");
        W(gate_proj,     "mlp.gate_proj.weight");
        W(up_proj,       "mlp.up_proj.weight");
        W(down_proj,     "mlp.down_proj.weight");
#undef W
    }

    fprintf(stderr, "embedding: loaded %s (hidden=%d heads=%d/%d layers=%d vocab=%d)\n",
            model_dir, c->hidden_size, c->num_heads, c->num_kv_heads,
            c->num_layers, c->vocab_size);
    return m;
}

void embedding_model_free(embedding_model *model) {
    if (!model) return;
    free(model->layers);
    free_tokenizer(model->tok);
    free_token_map(model->tok_map);
    free_weights(model->wf);
    free(model);
}

embedding_state *embedding_state_create(const embedding_model *model) {
    const EmbConfig *c = &model->cfg;
    int S = c->max_seq_len;
    int D = c->hidden_size;
    int H = c->num_heads;
    int KVH = c->num_kv_heads;
    int HD = c->head_dim;
    int MLP = c->intermediate_size;

    embedding_state *st = (embedding_state *)calloc(1, sizeof(embedding_state));
    /* Transformer layer scratch */
    st->x        = (float *)malloc((size_t)S * D * sizeof(float));
    st->tmp      = (float *)malloc((size_t)S * D * sizeof(float));
    st->attn     = (float *)malloc((size_t)S * D * sizeof(float));
    st->mlp_buf  = (float *)malloc((size_t)S * D * sizeof(float));
    /* Attention scratch */
    st->q        = (float *)malloc((size_t)S * H * HD * sizeof(float));
    st->k        = (float *)malloc((size_t)S * KVH * HD * sizeof(float));
    st->v        = (float *)malloc((size_t)S * KVH * HD * sizeof(float));
    st->attn_out = (float *)malloc((size_t)S * H * HD * sizeof(float));
    st->scores   = (float *)malloc((size_t)S * sizeof(float));
    /* MLP scratch */
    st->gate     = (float *)malloc((size_t)S * MLP * sizeof(float));
    st->up       = (float *)malloc((size_t)S * MLP * sizeof(float));
    /* Projection scratch */
    st->pooled   = (float *)malloc((size_t)D * sizeof(float));
    st->d1       = (float *)malloc((size_t)c->dense1_out * sizeof(float));
    st->d2       = (float *)malloc((size_t)c->dense2_out * sizeof(float));
    /* Tokenizer scratch */
    st->tokens   = (int *)malloc((size_t)S * sizeof(int));
    return st;
}

void embedding_state_free(embedding_state *state) {
    if (!state) return;
    free(state->x); free(state->tmp); free(state->attn); free(state->mlp_buf);
    free(state->q); free(state->k); free(state->v);
    free(state->attn_out); free(state->scores);
    free(state->gate); free(state->up);
    free(state->pooled); free(state->d1); free(state->d2);
    free(state->tokens);
    free(state);
}

int embedding_model_embed(const embedding_model *model,
                          embedding_state *state,
                          const char *text,
                          float *out, int out_dim) {
    if (!model || !state || !text || !out || out_dim <= 0) return -1;

    const EmbConfig *c = &model->cfg;
    int D = c->hidden_size;

    /* Tokenize */
    int seq = tokenize(model->tok_map, text, state->tokens, c->max_seq_len);
    if (seq <= 2) return -1; /* only BOS+EOS, no real tokens */

    /* Transformer forward (uses state scratch buffers) */
    float *hidden = emb_forward(model, state, state->tokens, seq);

    /* Mean pool → [D] */
    mean_pool(state->pooled, hidden, seq, D);

    /* Dense projections: D → dense1_out → dense2_out */
    matmul(state->d1, state->pooled, model->dense1_w, 1, D, c->dense1_out);
    matmul(state->d2, state->d1, model->dense2_w, 1, c->dense1_out, c->dense2_out);

    /* L2 normalize */
    l2_normalize(state->d2, c->dense2_out);

    /* Copy to output (truncate if out_dim < dense2_out) */
    int copy_dim = out_dim < c->dense2_out ? out_dim : c->dense2_out;
    memcpy(out, state->d2, copy_dim * sizeof(float));

    return 0;
}

float embedding_cosine_similarity(const float *a, const float *b, int dim) {
    /* For L2-normalized vectors, cosine similarity = dot product */
    return vec_dot(a, b, dim);
}

/* ───────────────── Public API: Intent Recognizer ───────────────── */

intent_recognizer *intent_recognizer_create(const embedding_model *model,
                                            embedding_state *state,
                                            float threshold) {
    intent_recognizer *ir = (intent_recognizer *)calloc(1, sizeof(intent_recognizer));
    ir->model = model;
    ir->reg_state = state;
    ir->threshold = threshold;
    ir->cap_intents = 16;
    ir->intents = (Intent *)calloc(ir->cap_intents, sizeof(Intent));
    return ir;
}

int intent_recognizer_register(intent_recognizer *ir,
                               const char *trigger_phrase,
                               intent_callback callback,
                               void *user_data) {
    /* Check if already registered */
    for (int i = 0; i < ir->num_intents; i++) {
        if (strcmp(ir->intents[i].trigger, trigger_phrase) == 0) {
            ir->intents[i].callback = callback;
            ir->intents[i].user_data = user_data;
            /* Recompute embedding */
            embedding_model_embed(ir->model, ir->reg_state, trigger_phrase,
                                  ir->intents[i].embedding, EMB_DIM);
            return 0;
        }
    }

    /* Grow array if needed */
    if (ir->num_intents >= ir->cap_intents) {
        ir->cap_intents *= 2;
        ir->intents = (Intent *)realloc(ir->intents,
                                        ir->cap_intents * sizeof(Intent));
    }

    Intent *it = &ir->intents[ir->num_intents];
    it->trigger = strdup(trigger_phrase);
    it->embedding = (float *)malloc(EMB_DIM * sizeof(float));
    it->callback = callback;
    it->user_data = user_data;

    if (embedding_model_embed(ir->model, ir->reg_state, trigger_phrase,
                              it->embedding, EMB_DIM) != 0) {
        free(it->trigger); free(it->embedding);
        return -1;
    }

    ir->num_intents++;
    return 0;
}

int intent_recognizer_unregister(intent_recognizer *ir,
                                 const char *trigger_phrase) {
    for (int i = 0; i < ir->num_intents; i++) {
        if (strcmp(ir->intents[i].trigger, trigger_phrase) == 0) {
            free(ir->intents[i].trigger);
            free(ir->intents[i].embedding);
            /* Shift remaining */
            for (int j = i; j < ir->num_intents - 1; j++)
                ir->intents[j] = ir->intents[j+1];
            ir->num_intents--;
            return 0;
        }
    }
    return -1;
}

int intent_recognizer_process(const intent_recognizer *ir,
                              const embedding_model *model,
                              embedding_state *state,
                              const char *utterance) {
    if (ir->num_intents == 0) return 0;

    float emb[EMB_DIM];
    if (embedding_model_embed(model, state, utterance, emb, EMB_DIM) != 0) return 0;

    float best_sim = -1.0f;
    int best_idx = -1;
    for (int i = 0; i < ir->num_intents; i++) {
        float sim = embedding_cosine_similarity(emb, ir->intents[i].embedding, EMB_DIM);
        if (sim > best_sim) { best_sim = sim; best_idx = i; }
    }

    if (best_idx >= 0 && best_sim >= ir->threshold) {
        Intent *it = &ir->intents[best_idx];
        it->callback(it->trigger, utterance, best_sim, it->user_data);
        return 1;
    }
    return 0;
}

void intent_recognizer_set_threshold(intent_recognizer *ir, float threshold) {
    ir->threshold = threshold;
}

float intent_recognizer_get_threshold(const intent_recognizer *ir) {
    return ir->threshold;
}

void intent_recognizer_free(intent_recognizer *ir) {
    if (!ir) return;
    for (int i = 0; i < ir->num_intents; i++) {
        free(ir->intents[i].trigger);
        free(ir->intents[i].embedding);
    }
    free(ir->intents);
    free(ir);
}
