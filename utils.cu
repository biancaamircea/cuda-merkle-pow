#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

#define HASH_HEX_LEN (SHA256_HASH_SIZE - 1)

#define MERKLE_THREADS 256
#define NONCE_THREADS 256
#define NONCE_BATCH_SIZE (1ULL << 16)

#define WARMUP_BLOCKS 512
#define WARMUP_THREADS 256
#define WARMUP_CYCLES 100000000ULL
#define WARMUP_ROUNDS 10

#define PREALLOC_TRANSACTIONS_BYTES (64ULL * 1024ULL * 1024ULL)
#define PREALLOC_HASHES_BYTES (16ULL * 1024ULL * 1024ULL)

static BYTE *g_d_transactions = NULL;
static BYTE *g_d_hashes_a = NULL;
static BYTE *g_d_hashes_b = NULL;

static size_t g_transactions_cap = 0;
static size_t g_hashes_a_cap = 0;
static size_t g_hashes_b_cap = 0;

static uint32_t *g_d_best_nonce = NULL;
static int *g_d_found = NULL;

static void ensure_byte_buffer(BYTE **ptr, size_t *capacity, size_t needed) {
    if (*ptr != NULL && *capacity >= needed) {
        return;
    }

    if (*ptr != NULL) {
        CUDA_CHECK(cudaFree(*ptr));
    }

    CUDA_CHECK(cudaMalloc((void **)ptr, needed));
    *capacity = needed;
}

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ __forceinline__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }

    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src) {
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src) {
    while (*dest != '\0') {
        dest++;
    }

    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }

    *dest = '\0';
}

__host__ __device__ void sha256_to_hex(BYTE *buf, BYTE *output) {
    const char hex_chars[] = "0123456789abcdef";

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
    }

    output[SHA256_BLOCK_SIZE * 2] = '\0';
}

__host__ __device__ void apply_sha256_len(const BYTE *input, size_t input_length, BYTE *output) {
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    sha256_to_hex(buf, output);
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    apply_sha256_len(input, d_strlen((const char *)input), output);
}

__device__ void apply_sha256_two_parts(
    const BYTE *a,
    size_t len_a,
    const BYTE *b,
    size_t len_b,
    BYTE *output
) {
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];

    sha256_init(&ctx);
    sha256_update(&ctx, a, len_a);
    sha256_update(&ctx, b, len_b);
    sha256_final(&ctx, buf);

    sha256_to_hex(buf, output);
}

__device__ void apply_sha256_ctx_suffix(
    SHA256_CTX prefix_ctx,
    const BYTE *suffix,
    size_t suffix_len,
    BYTE *output
) {
    BYTE buf[SHA256_BLOCK_SIZE];

    sha256_update(&prefix_ctx, suffix, suffix_len);
    sha256_final(&prefix_ctx, buf);

    sha256_to_hex(buf, output);
}

__device__ __forceinline__ int sha256_ctx_suffix_has_zero_prefix(
    SHA256_CTX prefix_ctx,
    const BYTE *suffix,
    size_t suffix_len,
    int zero_nibbles
) {
    BYTE buf[SHA256_BLOCK_SIZE];

    sha256_update(&prefix_ctx, suffix, suffix_len);
    sha256_final(&prefix_ctx, buf);

    int full_zero_bytes = zero_nibbles / 2;

    for (int i = 0; i < full_zero_bytes; i++) {
        if (buf[i] != 0) {
            return 0;
        }
    }

    if (zero_nibbles & 1) {
        if ((buf[full_zero_bytes] >> 4) != 0) {
            return 0;
        }
    }

    return 1;
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1;
        } else if (hash1[i] > hash2[i]) {
            return 1;
        }
    }

    return 0;
}

__global__ void hash_transactions_kernel(
    int transaction_size,
    BYTE *transactions,
    int n,
    BYTE *hashes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        apply_sha256_len(
            transactions + idx * transaction_size,
            (size_t)(transaction_size - 1),
            hashes + idx * SHA256_HASH_SIZE
        );
    }
}

__global__ void merkle_level_kernel(
    BYTE *in_hashes,
    BYTE *out_hashes,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int new_n = (n + 1) / 2;

    if (idx < new_n) {
        BYTE *left = in_hashes + (2 * idx) * SHA256_HASH_SIZE;
        BYTE *right;

        if (2 * idx + 1 < n) {
            right = in_hashes + (2 * idx + 1) * SHA256_HASH_SIZE;
        } else {
            right = left;
        }

        apply_sha256_two_parts(
            left,
            HASH_HEX_LEN,
            right,
            HASH_HEX_LEN,
            out_hashes + idx * SHA256_HASH_SIZE
        );
    }
}

__global__ void find_nonce_kernel(
    SHA256_CTX prefix_ctx,
    uint64_t start_nonce,
    uint64_t count,
    uint32_t *best_nonce,
    int *found,
    int zero_nibbles
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count) {
        return;
    }

    uint64_t nonce64 = start_nonce + tid;

    if (nonce64 > UINT32_MAX) {
        return;
    }

    uint32_t nonce = (uint32_t)nonce64;

    char nonce_string[NONCE_SIZE];
    int nonce_len = intToString((uint64_t)nonce, nonce_string);

    if (sha256_ctx_suffix_has_zero_prefix(
            prefix_ctx,
            (BYTE *)nonce_string,
            nonce_len,
            zero_nibbles
        )) {
        atomicMin(best_nonce, nonce);
        atomicExch(found, 1);
    }
}

// TODO 1: Implement this function in CUDA
void construct_merkle_root(
    int transaction_size,
    BYTE *transactions,
    int max_transactions_in_a_block,
    int n,
    BYTE merkle_root[SHA256_HASH_SIZE]
) {
    (void)max_transactions_in_a_block;

    if (n <= 0) {
        merkle_root[0] = '\0';
        return;
    }

    size_t transactions_bytes = (size_t)n * transaction_size;
    size_t hashes_bytes = (size_t)n * SHA256_HASH_SIZE;

    ensure_byte_buffer(&g_d_transactions, &g_transactions_cap, transactions_bytes);
    ensure_byte_buffer(&g_d_hashes_a, &g_hashes_a_cap, hashes_bytes);
    ensure_byte_buffer(&g_d_hashes_b, &g_hashes_b_cap, hashes_bytes);

    CUDA_CHECK(cudaMemcpy(
        g_d_transactions,
        transactions,
        transactions_bytes,
        cudaMemcpyHostToDevice
    ));

    int threads = MERKLE_THREADS;
    int blocks = (n + threads - 1) / threads;

    // Compute the SHA256 hash for each transaction
    hash_transactions_kernel<<<blocks, threads>>>(
        transaction_size,
        g_d_transactions,
        n,
        g_d_hashes_a
    );
    CUDA_CHECK(cudaGetLastError());

    // Build the Merkle tree
    int current_n = n;
    BYTE *in_hashes = g_d_hashes_a;
    BYTE *out_hashes = g_d_hashes_b;

    while (current_n > 1) {
        int new_n = (current_n + 1) / 2;
        int level_blocks = (new_n + threads - 1) / threads;

        merkle_level_kernel<<<level_blocks, threads>>>(
            in_hashes,
            out_hashes,
            current_n
        );
        CUDA_CHECK(cudaGetLastError());

        BYTE *tmp = in_hashes;
        in_hashes = out_hashes;
        out_hashes = tmp;

        current_n = new_n;
    }

    CUDA_CHECK(cudaMemcpy(
        merkle_root,
        in_hashes,
        SHA256_HASH_SIZE,
        cudaMemcpyDeviceToHost
    ));
}

// TODO 2: Implement this function in CUDA
int find_nonce(
    BYTE *difficulty,
    uint32_t max_nonce,
    BYTE *block_content,
    size_t current_length,
    BYTE *block_hash,
    uint32_t *valid_nonce
) {
    if (g_d_best_nonce == NULL) {
        CUDA_CHECK(cudaMalloc((void **)&g_d_best_nonce, sizeof(uint32_t)));
    }

    if (g_d_found == NULL) {
        CUDA_CHECK(cudaMalloc((void **)&g_d_found, sizeof(int)));
    }

    SHA256_CTX prefix_ctx;
    sha256_init(&prefix_ctx);
    sha256_update(&prefix_ctx, block_content, current_length);

    int zero_nibbles = 0;
    while (zero_nibbles < SHA256_HASH_SIZE - 1 && difficulty[zero_nibbles] == '0') {
        zero_nibbles++;
    }

    const int threads = NONCE_THREADS;
    const uint64_t BATCH_SIZE = NONCE_BATCH_SIZE;

    uint64_t total = (uint64_t)max_nonce + 1ULL;
    uint64_t start_nonce = 0;

    while (start_nonce < total) {
        uint64_t count = BATCH_SIZE;

        if (start_nonce + count > total) {
            count = total - start_nonce;
        }

        int blocks = (int)((count + threads - 1) / threads);

        CUDA_CHECK(cudaMemset(
            g_d_best_nonce,
            0xFF,
            sizeof(uint32_t)
        ));

        CUDA_CHECK(cudaMemset(
            g_d_found,
            0,
            sizeof(int)
        ));

        find_nonce_kernel<<<blocks, threads>>>(
            prefix_ctx,
            start_nonce,
            count,
            g_d_best_nonce,
            g_d_found,
            zero_nibbles
        );
        CUDA_CHECK(cudaGetLastError());

        int host_found = 0;

        CUDA_CHECK(cudaMemcpy(
            &host_found,
            g_d_found,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        if (host_found) {
            uint32_t host_best = UINT32_MAX;

            CUDA_CHECK(cudaMemcpy(
                &host_best,
                g_d_best_nonce,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost
            ));

            *valid_nonce = host_best;

            char nonce_string[NONCE_SIZE];
            sprintf(nonce_string, "%u", host_best);
            strcpy((char *)block_content + current_length, nonce_string);
            apply_sha256(block_content, block_hash);

            return 0;
        }

        start_nonce += count;
    }

    return 1;
}

__global__ void dummy_kernel() {}

__global__ void clock_warmup_kernel(uint32_t *sink) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = (uint32_t)(tid + 1);

    unsigned long long start = clock64();

    while (clock64() - start < WARMUP_CYCLES) {
        x ^= x << 7;
        x += 12345u;
        x ^= x >> 3;
        x += (uint32_t)threadIdx.x;
    }

    sink[tid] = x;
}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    CUDA_CHECK(cudaMalloc((void **)&dummy_data, 256));
    dummy_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dummy_data));

    ensure_byte_buffer(&g_d_transactions, &g_transactions_cap, PREALLOC_TRANSACTIONS_BYTES);
    ensure_byte_buffer(&g_d_hashes_a, &g_hashes_a_cap, PREALLOC_HASHES_BYTES);
    ensure_byte_buffer(&g_d_hashes_b, &g_hashes_b_cap, PREALLOC_HASHES_BYTES);

    if (g_d_best_nonce == NULL) {
        CUDA_CHECK(cudaMalloc((void **)&g_d_best_nonce, sizeof(uint32_t)));
    }

    if (g_d_found == NULL) {
        CUDA_CHECK(cudaMalloc((void **)&g_d_found, sizeof(int)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    
    uint32_t *warm_sink = (uint32_t *)g_d_hashes_a;

    for (int i = 0; i < WARMUP_ROUNDS; i++) {
        clock_warmup_kernel<<<WARMUP_BLOCKS, WARMUP_THREADS>>>(warm_sink);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}