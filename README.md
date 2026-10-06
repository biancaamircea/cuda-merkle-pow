# CUDA Merkle Tree and Proof of Work

Academic project by Bianca-Anastasia Mircea, May 2026.

CUDA implementation of transaction hashing, level-by-level Merkle reduction, and a batched nonce search for a simplified Proof-of-Work assignment. Uses reusable GPU buffers, a cached SHA-256 prefix context, and atomic selection of the smallest valid nonce in each batch.

## Files

- `utils.cu`: submitted CUDA implementation.
- `docs/original_README.md`: original implementation explanation and AI assistance disclosure.

## Reproducibility

This is a course submission fragment. The supplied archive does not include the course skeleton, `utils.h`, SHA-256 support files, Makefile or test inputs. The commands in the original README assume those external files are present. A CUDA-capable NVIDIA GPU and matching CUDA toolchain are required. No new performance measurements were produced during portfolio preparation.
