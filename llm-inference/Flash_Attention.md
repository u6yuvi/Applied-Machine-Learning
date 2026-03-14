# Flash Attention

## 1. The Core Problem: The "Memory Wall"

**The Bottleneck:** Standard attention is memory-bound, not compute-bound. The GPU spends more time moving data between HBM and SRAM than actually performing math.

- **Standard Attention \(O(N^2)\):** Requires building and storing the full \(N \times N\) attention matrix (\(S\) and \(P\)) in HBM.
- **Flash Attention \(O(N)\):** Reduces memory complexity to linear by never materializing the full \(N \times N\) matrix in HBM.

**Key Insight:** It is computationally cheaper to recompute values on-chip (SRAM) than to read/write them from VRAM (HBM).

---

## 2. GPU Memory Hierarchy (The Hardware Context)

| Layer | Speed | Size | Flash Attention Usage |
| --- | --- | --- | --- |
| Registers | Fastest | Tiny | Immediate scalar/vector math. |
| SRAM | Very Fast | MBs | Where the "Tiles" are processed. |
| HBM (VRAM) | Slow | GBs | Where \(Q\), \(K\), \(V\), and \(O\) are stored long-term. |

---

## 3. The Three Pillars of Flash Attention

### Tiling

Blocks of \(Q\), \(K\), and \(V\) are loaded into SRAM. Instead of one giant matrix, we compute attention "tile-by-tile."

### Online Softmax

Softmax usually requires the full row to find the maximum for numerical stability. Flash Attention uses a running maximum (\(m\)) and a running sum of exponentials (\(l\)).

**Correction Factor:** If a new tile has a higher max (\(m_{new}\)), the previous partial result is rescaled by \(e^{(m_{old} - m_{new})}\).

### Recomputation

To save memory, the \(N \times N\) matrix is not stored during the forward pass. During the backward pass (gradient calculation), the algorithm re-tiles and re-computes the attention matrix on the fly.

---

## 4. Algorithm Evolution (FA1 → FA4)

- **Flash Attention 1:** Introduced Tiling + Recomputation. Reduced HBM traffic.

- **Flash Attention 2:**
  - Parallelized over the sequence length dimension.
  - Optimized the loop order (Load Query tile once, stream all K/V tiles).
  - 80%+ Tensor Core utilization.

- **Flash Attention 3 (Hopper/H100):**
  - **Asynchronous Execution:** Uses the H100 "Tensor Memory Accelerator" (TMA) to fetch data while the GPU is still calculating.
  - **Warp Specialization:** Separate groups of threads for data movement vs. math.

- **Flash Attention 4 (Blackwell/B200):**
  - **Polynomial Approximation:** Replaces slow Special Function Units (SFU) for \(\exp(x)\) with faster polynomial math on CUDA cores.
  - Achieves Petaflop scale throughput.

---

## 5. Flash Attention vs. KV Cache

- **KV Cache:** Optimizes Inference (Decoding). It saves past \(K\) and \(V\) vectors so we don't re-process the whole prompt for every new token.

- **Flash Attention:** Optimizes the Kernel. It makes the actual math of \(Q \times K\) faster and less memory-intensive.

- **In production:** vLLM uses Flash Attention kernels to process the blocks stored in the PagedAttention KV Cache.

---

## 6. Practitioner's Corner: Tuning for vLLM/Deployment

When deploying models (Llama 3, Mistral) in vLLM using Flash Attention, tune these hyperparameters:

- **--max-model-len:** Because FA is \(O(N)\) memory, you can push this much higher (e.g., 32k or 128k) than standard attention. However, higher limits reserve more memory for the KV Cache.

- **--gpu-memory-utilization (Default: 0.90):** Determines how much VRAM is allocated to the KV Cache. If you get "Out of Memory" (OOM) during long context processing, check if Flash Attention is actually enabled (requires Triton/CUDA).

- **block_size (PagedAttention Integration):** vLLM uses blocks (usually 16 or 32 tokens). While FA handles the internal tiling, matching the block_size to the hardware's alignment (usually 128-bit) ensures optimal IO.

- **dtype (FP16 vs. BF16 vs. FP8):** BF16 is preferred for Flash Attention to avoid the overflow issues common in FP16. FP8 is only supported in Flash Attention 3+ (H100/B200) and provides a massive speedup in vLLM for high-throughput scenarios.

---

## 7. Common "Gotchas"

- **Is it an approximation?** No. It is an exact algorithm. The output is mathematically identical to standard attention (within floating-point rounding error).

- **Does it help for short sequences?** No. For \(N < 512\), the overhead of tiling might even make it slightly slower. It shines as \(N\) grows (\(>2k\) tokens).

- **What is saved for the backward pass?** Only the softmax normalization statistics (\(m\) and \(l\)) and the output \(O\). The \(N \times N\) attention weights are discarded.

- **What is the "IO-Awareness" in one sentence?** It is the strategy of accounting for the speed difference between memory levels (HBM vs. SRAM) to minimize data movement.
