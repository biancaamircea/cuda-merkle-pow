# Computer Systems Architecture Assignment 2 -- CUDA Merkle Root and Proof of Work

**Student:** Mircea Bianca-Anastasia  
**Group:** 333CC  
**Files included in the original submission archive:** `utils.cu`, `README.md`

## 1. Overview

This assignment implements two expensive stages of a simplified block-mining process on the GPU:

- computing the Merkle root of the block's transactions;
- finding a valid Proof-of-Work nonce.

The implementation in `utils.cu` uses CUDA to distribute work across threads. The CPU coordinates calls, transfers data and writes the final result. The GPU performs repetitive hashing.

The solution uses three main ideas:

1. Transactions are hashed in parallel.
2. The Merkle tree is built level by level using two alternating buffers.
3. Nonces are tested in batches, and a valid result is selected using atomic operations.

## 2. Key code elements

The implementation uses these constants:

```c
#define MERKLE_THREADS 256
#define NONCE_THREADS 256
#define NONCE_BATCH_SIZE (1ULL << 16)
```

Both the Merkle and nonce kernels use 256 threads per block. The nonce search is split into batches of 2^16 values to limit each kernel launch and allow periodic host-side checks.

GPU memory is stored in global buffers:

```c
static BYTE *g_d_transactions = NULL;
static BYTE *g_d_hashes_a = NULL;
static BYTE *g_d_hashes_b = NULL;

static uint32_t *g_d_best_nonce = NULL;
static int *g_d_found = NULL;
```

These buffers are reused across calls to avoid repeated allocations.

## 3. Memory management

The allocation helper is:

```c
static void ensure_byte_buffer(BYTE **ptr, size_t *capacity, size_t needed)
```

It checks whether the existing buffer has enough capacity. If so, it reuses it. Otherwise, it frees the old buffer and allocates a replacement with `cudaMalloc`.

Both `construct_merkle_root` and `find_nonce` may be called for multiple blocks. Reusing memory avoids frequent `cudaMalloc` and `cudaFree` calls.

All CUDA calls are checked using `CUDA_CHECK` so errors can be detected immediately.

## 4. Computing the Merkle root

### 4.1. Initial transaction hashes

`construct_merkle_root` first copies transactions to the GPU:

```c
cudaMemcpy(g_d_transactions, transactions, transactions_bytes, cudaMemcpyHostToDevice)
```

It then launches:

```c
hash_transactions_kernel<<<blocks, threads>>>(...)
```

Each thread hashes one transaction. Its index is derived from `blockIdx`, `blockDim` and `threadIdx`.

The hash uses `transaction_size - 1` bytes because the transaction size includes the terminating null character, which must not be hashed.

### 4.2. Level reduction

After hashing the transactions, the array is reduced until one hash remains:

```c
merkle_level_kernel<<<level_blocks, threads>>>(...)
```

Each thread produces one entry in the next level by processing two hashes:

```c
left  = in_hashes + (2 * idx) * SHA256_HASH_SIZE;
right = in_hashes + (2 * idx + 1) * SHA256_HASH_SIZE;
```

If the second hash does not exist, `right` is set to `left`. This implements the Merkle rule of duplicating the last hash when a level has an odd number of nodes.

### 4.3. Logical concatenation

Rather than copying both hashes into a separate concatenation buffer, `apply_sha256_two_parts` updates the same SHA-256 context with each part:

```c
sha256_update(&ctx, a, len_a);
sha256_update(&ctx, b, len_b);
```

This produces the same result as hashing the concatenated input while avoiding an intermediate copy.

### 4.4. Alternating buffers

The tree reduction uses `g_d_hashes_a` and `g_d_hashes_b`. Each level reads from one buffer and writes to the other. Their pointers are swapped after each level:

```c
BYTE *tmp = in_hashes;
in_hashes = out_hashes;
out_hashes = tmp;
```

This avoids copying intermediate levels to the CPU.

When `current_n` reaches 1, the remaining hash is copied to the host's `merkle_root`.

## 5. Nonce search

### 5.1. Batched search

`find_nonce` searches from 0 to `max_nonce` in batches:

```c
const uint64_t BATCH_SIZE = NONCE_BATCH_SIZE;
```

For each batch, the required CUDA block count is calculated and `find_nonce_kernel` is launched.

### 5.2. SHA-256 prefix context

Before launching the kernel, the fixed portion of the block is processed:

```c
SHA256_CTX prefix_ctx;
sha256_init(&prefix_ctx);
sha256_update(&prefix_ctx, block_content, current_length);
```

The context is passed to the kernel. Each thread copies it and appends only its nonce, avoiding repeated processing of the fixed prefix.

### 5.3. Nonce conversion

The nonce must be appended to the block content as a string. The GPU uses:

```c
__device__ __forceinline__ int intToString(uint64_t num, char* out)
```

This converts the number to decimal text. A custom function is used instead of `sprintf` to avoid its cost across many CUDA threads.

### 5.4. Difficulty check

The kernel does not generate and compare full hexadecimal strings for every candidate. Instead:

```c
sha256_ctx_suffix_has_zero_prefix(...)
```

computes the binary digest and checks the required number of leading hexadecimal zeros.

An even number of zeros is checked as whole zero bytes. An odd number additionally requires the upper nibble of the next byte to be zero. This avoids converting every digest to text.

### 5.5. Selecting the result

Several threads may find valid nonces in the same batch. The smallest is retained using:

```c
atomicMin(best_nonce, nonce);
```

A separate flag is set:

```c
atomicExch(found, 1);
```

After each kernel, the host copies `found`. If it is 1, it copies `best_nonce`, the smallest valid nonce in that batch.

Batches are processed in increasing order, so the first successful batch is sufficient to select the final nonce.

The selected nonce is appended to `block_content`, and the final block hash is recalculated on the host.

## 6. Helper functions

Several helpers are retained or adapted for host/device use:

- `d_strlen`: a simple `strlen` implementation.
- `d_strcpy`: device string copying.
- `d_strcat`: device string concatenation.
- `sha256_to_hex`: binary-digest-to-hexadecimal conversion.
- `apply_sha256_len`: SHA-256 for an input of known length.
- `apply_sha256`: SHA-256 for a null-terminated string.

These preserve the hash format used by the CPU implementation.

## 7. GPU warm-up

`warm_up_gpu` has two purposes:

1. Initialize the CUDA context by launching a simple kernel.
2. Preallocate the main buffers.

It also launches `clock_warmup_kernel`, which executes a GPU loop to make the GPU active before timing important functions.

Preallocation uses fixed sizes:

```c
#define PREALLOC_TRANSACTIONS_BYTES (64ULL * 1024ULL * 1024ULL)
#define PREALLOC_HASHES_BYTES (16ULL * 1024ULL * 1024ULL)
```

For typical tests, the memory is therefore available before block processing begins.

## 8. Correctness

The implementation follows these rules:

- Transactions are hashed without their terminating null character.
- Merkle hashes are combined in pairs.
- The final hash is duplicated when a level has an odd number of nodes.
- Reduction continues until one hash remains.
- Nonce batches are generated in increasing order.
- `atomicMin` retains the smallest valid nonce within a batch.
- `found` indicates whether the current batch contains a valid result.
- The block hash is recalculated on the host after nonce selection.

## 9. Building and running

Build:

```bash
make
```

Run:

```bash
make run TEST=test1
make run TEST=test2
make run TEST=test3
make run TEST=test4
```

Clean:

```bash
make clean
```

Final tests must run on the infrastructure specified in the assignment because execution times depend on the GPU.

## 10. Performance observations

Merkle parallelization is most useful for blocks with many transactions. Initial hashes are independent, and each tree level reduces the number of elements.

Nonce-search performance depends on difficulty and the position of the first valid nonce. An early result requires few batches; a later result requires more.

Batch size is a tradeoff. Smaller batches allow more frequent host-side checks but require more kernel launches and small transfers. Larger batches reduce launch overhead but may perform more work after a valid nonce has already been found within a batch.

This implementation uses 2^16 nonces per batch.

## 11. Limitations and possible improvements

Possible improvements include:

- Testing different `NONCE_BATCH_SIZE` values, since the best size depends on the workload and GPU.
- Replacing the `found` flag with a direct `best_nonce` check to reduce one `cudaMemset` and one transfer per batch, provided the no-result sentinel is handled correctly. The current version explicitly separates result presence from the minimum nonce value.
- Processing very small Merkle levels on the CPU or combining steps to reduce kernel-launch overhead when few nodes remain.

## 12. LLM prompts used

**Tool:** ChatGPT -- GPT-5.5 Thinking  
**Purpose:** clarifying CUDA concepts, understanding implementation tradeoffs and drafting documentation.

The prompts below are English translations of those recorded in the original submission.

### Prompt 1

**Question:**

> How can I construct a Merkle root on the GPU from an array of transactions when the last hash must be duplicated if the number of elements is odd?

**Response summary:** First hash transactions in parallel, then construct the tree level by level. A thread processes one pair of hashes; an incomplete pair uses the same hash twice.

**Usefulness:** Helped organize `merkle_level_kernel` and validate the last-hash duplication rule.

### Prompt 2

**Question:**

> Why are two device buffers used to construct the Merkle tree, and how are their pointers swapped between levels?

**Response summary:** With ping-pong buffers, the current level is read from one buffer and the next is written to the other. Swapping pointers avoids new allocations at each step and intermediate CPU copies.

**Usefulness:** Clarified why `g_d_hashes_a` and `g_d_hashes_b` are sufficient for the entire tree.

### Prompt 3

**Question:**

> How can I search GPU nonces in batches and determine whether a batch contains a valid nonce?

**Response summary:** Each thread tests `start_nonce + tid`. A successful thread sets a flag with `atomicExch`; `atomicMin` retains the smallest valid nonce.

**Usefulness:** Helped structure `find_nonce_kernel`, which uses both `g_d_found` and `g_d_best_nonce`.

### Prompt 4

**Question:**

> Why is it more efficient to initialize the SHA-256 context for the block prefix once and append only the nonce in the kernel?

**Response summary:** The fixed block prefix is identical for every nonce. Processing it once allows each thread to handle only the variable suffix, reducing repeated work.

**Usefulness:** Clarified the optimization in `find_nonce`, where `prefix_ctx` is prepared on the host and passed to the kernel.

### Prompt 5

**Question:**

> What should a README explain for a CUDA Merkle-root and Proof-of-Work assignment so that the implementation is clear during evaluation?

**Response summary:** Explain the Merkle and nonce flows separately, GPU allocations, atomic operations, correctness checks and testing. Include manual decisions and limitations.

**Usefulness:** Helped structure this README and make its explanations easier to follow.

## 13. Manual decisions

The final implementation includes these manually selected choices:

- 256 threads per block for the main kernels.
- Global buffers to avoid repeated allocations.
- Batched nonce searching.
- A `found` flag to detect whether a batch contains a result.
- `atomicMin` to select the smallest valid nonce.
- A custom device-side nonce conversion function instead of `sprintf`.
- Host-side recalculation of the final block hash.
- A warm-up function to initialize and prepare the GPU.

## 14. Conclusion

The solution moves repeated transaction hashing, Merkle reduction and nonce testing onto the GPU. Global buffers, level-wise reduction and batched searching reduce serial CPU work.

The implementation is designed to be understandable, verifiable and compatible with the assignment requirements. Exact performance must be evaluated on the specified infrastructure.
