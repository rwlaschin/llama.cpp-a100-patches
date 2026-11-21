# llama.cpp A100 patches

The patch stack behind our LinkedIn series on running 32 prompts at once on one A100 with llama.cpp. It applies to
llama.cpp commit [`69ee2ecc`](https://github.com/ggml-org/llama.cpp/commit/69ee2ecc0150ba7d1204a7f7393dfd651569db8c)
and builds one binary that runs on both the A100 and the L4.

```bash
./build.sh                     # clones llama.cpp into ./llama.cpp, applies the patches, builds llama-server
./build.sh ~/src/llama.cpp "80;89"
```

At run time, set `GGML_CUDA_MMQ_Q8_CACHE=20` to pin the quantize-once buffer at 20 MiB (see `it3a` below).

## Results (A100, same commit, stock vs this stack)

| | stock | patched |
|---|---|---|
| prompt reading, 1,746 tokens | 1,410 tok/s | 2,842 tok/s (+102%) |
| prompt reading, 512 tokens | 1,384 tok/s | 2,208 tok/s (+59%) |
| decode, 32 sequences at once (A100 40 GB) | — | 585 tok/s |
| perplexity, 4 chunks | 4.3354 / 3.4234 / 2.4075 / 1.9471 | 4.2851 / 3.3957 / 2.3945 / 1.9389 |

On the L4 the stack matches stock on prompt speed (871 vs 869 tok/s at 512 tokens) with bit-identical perplexity.

## The patches, in apply order

| patch | what it does | where it runs |
|---|---|---|
| `gdn` | adds the A100 (GA100) to the gated-delta-net tensor-core allowlist | A100 |
| `sampler` | a sampler change drops cached graphs instead of rebuilding the scheduler (removes a ~330 ms stall per new request) | all |
| `yonce` | `GGML_CUDA_MMQ_Y_ONCE`: decode tiles load the whole activation tile per step, two syncs instead of four | all |
| `skeff` | `GGML_CUDA_MMQ_SK_EFF`: makes the stream-k tiling-efficiency threshold (default 90) an env setting | all |
| `rsview` | **upstream [PR #30139](https://github.com/ggml-org/llama.cpp/pull/30139), included as-is. Not our work.** | all |
| `st` | `GGML_CUDA_MMQ_STAGE`: copies the next step's weights into shared memory with cp.async. Reimplements the idea of upstream [PR #28572](https://github.com/ggml-org/llama.cpp/pull/28572) for every quant type | A100 (via `l4gate`) |
| `jcap` | `GGML_CUDA_MMQ_JMAX`: caps the widest tile tried (default 128) | all |
| `pad3` | `GGML_CUDA_MMQ_I2F_EXACT`: replaces the A100's quarter-rate int-to-float convert with an integer add and a float add; also stage padding/linear-copy options, off by default | I2F: A100 (via `l4gate`) |
| `it2` | merges each gated-delta-net layer's beta and alpha weights into one matmul at load | all |
| `it3a` | `GGML_CUDA_MMQ_Q8_CACHE`: quantize an activation once and reuse it for q, k and v, in a self-sizing ring. Reimplements the idea of upstream [PR #18538](https://github.com/ggml-org/llama.cpp/pull/18538) | all |
| `it56s` | `GGML_CUDA_MMQ_Y_ASYNC`: copies the next activation tile with cp.async; `STAGE_RING` (off) | with staging |
| `it7a` | `GGML_CUDA_MMQ_CUBLAS_ABOVE=512`: batches of 512+ columns go to cuBLAS fp16, smaller ones stay on MMQ | A100 |
| `cmake` | exposes the switches above as CMake options | all |
| `l4gate` | compiles `I2F_EXACT` and `STAGE` in for `__CUDA_ARCH__ == 800` only, at every site including host-side shared-memory sizing | — |
| `i64` | `GGML_CUDA_MMQ_I64`: decode tiles of 64 rows × 128 threads, two blocks per SM | A100 |
| `topk1` | radix top-k in 6 launches instead of 11 | all |

`skeff`, `gdn` and `sampler` each change one line of code that upstream introduced (PRs #22298, #29353, #18547); the
change is ours, the surrounding code is not.

## Tested

- CUDA archs 80 (A100) and 89 (L4). Arch 120 compiles but has never run on a card.
- `test-backend-ops`: all 2,517 `MUL_MAT` and 931 `MUL_MAT_ID` cases pass on both cards.
- Perplexity within 0.069 per chunk of the unpatched build.

## License

MIT, same as llama.cpp. `rsview.patch` belongs to the authors of PR #30139.

Claude Fable 5.1 and Claude Opus were used to help make these changes; a human reviewed every line.
