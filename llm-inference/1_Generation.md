# The KV Cache

## 1. The Core Concept: Why do we need it?

### The Problem

Transformers are stateless. To predict token \(t\), the model needs to calculate Attention across all \(t-1\) previous tokens. Without caching, you must feed the entire sequence through the model at every single step, recalculating the Key (K) and Value (V) vectors from scratch.

### The Solution (KV Cache)

We save the K and V vectors for previous tokens. For every new token generated, the input sequence length is exactly 1. We only calculate Q, K, and V for the new token, retrieve the past K and V from the cache, and update the cache.

---

## 2. The Complexity Win

KV cache doesn't make generation \(O(1)\), but it massively reduces redundant math.

- **Without Cache:** Generating the \(t\)-th token takes \(O(t^2)\) compute. Generating an \(N\)-token sequence takes \(O(N^3)\).
- **With Cache:** Computing K/V for the new token is \(O(1)\). Computing attention is \(O(t)\). Generating an \(N\)-token sequence drops to \(O(N^2)\).

**Why attention is \(O(t^2)\):** At each position, the model does attention: "this token looks at all other tokens (and itself)." So:

- Position 1 attends to: 1, 2, …, t → t keys/values
- Position 2 attends to: 1, 2, …, t → t keys/values
- …
- Position t attends to: 1, 2, …, t → t keys/values

That’s \(t \times t = t^2\) attention operations → **O(t²)**.

**Why \(O(N^3)\) without cache:** Step 1 costs \(O(1^2)\), step 2 costs \(O(2^2)\), …, step N costs \(O(N^2)\). Sum \(1^2 + 2^2 + \cdots + N^2 \approx N^3\).

**Why \(O(N^2)\) with cache:** Step 1 costs \(O(1)\), …, step N costs \(O(N)\). Sum \(1 + 2 + \cdots + N \approx N^2\).

**Takeaway:** Cache removes redoing full forward passes; we only pay for the new token plus attention over the cache.

|   | Per step | Total for N tokens |
| --- | --- | --- |
| **Without cache** | \(O(t^2)\) | \(O(N^3)\) |
| **With cache** | \(O(t)\) | \(O(N^2)\) |

---

## 3. The Memory Footprint Formula

Interviewers will ask you to estimate KV cache memory. Memorize this formula:

```
2 × Num_Layers × Batch_Size × Seq_Len × Hidden_Dim × Bytes_Per_Param
```

- **2:** Because we store both Keys and Values.
- **Num_Layers:** Every transformer layer has its own separate cache!
- **Bytes_Per_Param:** Usually 2 bytes (FP16 or BF16).

---

## 4. Step-by-Step Matrix Shapes (Batch=1, Heads=1, Dim=d)

How the tensors change shape as tokens are generated.

### Phase 1: Prefill (User sends a 3-token prompt)

- **Input:** `[1, 3, d]` (Batch=1, Tokens=3, Dim=d)
- **Compute Q, K, V:** `[1, 3, d]`
- **Initialize Cache:** Save K and V → K_cache and V_cache are both `[1, 3, d]`.
- **Output:** Token 4.

### Phase 2: Decode Step 1 (Feed Token 4 back in)

- **Input:** `[1, 1, d]` (Just the 1 new token!)
- **Compute new Q, K, V:** `[1, 1, d]`
- **Update Cache:** Append new K, V to cache → Cache shape becomes `[1, 4, d]`.
- **Attention Math:**
  - Q @ K_cache.T → `[1, 1, d]` @ `[1, d, 4]` = `[1, 1, 4]` (Attention scores)
  - Scores @ V_cache → `[1, 1, 4]` @ `[1, 4, d]` = `[1, 1, d]` (Final Output)
- **Output:** Token 5.

### Phase 3: Decode Step 2 (Feed Token 5 back in)

- **Input:** `[1, 1, d]`
- **Update Cache:** Cache shape grows to `[1, 5, d]`.
- **Attention Math:** `[1, 1, d]` @ `[1, d, 5]` → `[1, 1, 5]` @ `[1, 5, d]` = `[1, 1, d]`

---

## 5. Advanced Topics

### 5.1 Key LLM Serving Metrics: Prefill vs. Decode

To measure how effectively a system handles KV caching and generation, we split metrics by generation phase:

**The Prefill Phase (Compute-Bound)**

Processing the entire user prompt at once. The GPU math cores are maxed out doing giant matrix multiplications.

- **Time to First Token (TTFT):** The time it takes from sending the prompt to receiving the very first generated token. This metric represents the time spent in the Prefill phase processing the prompt and setting up the initial KV cache.

**The Decode Phase (Memory-Bound)**

Generating tokens one by one. The math is tiny (1 × d), but the GPU is bottlenecked by fetching the massive KV Cache from VRAM to compute cores.

- **Inter-Token Latency (ITL) / Time Per Output Token (TPOT):** The time it takes to generate each subsequent token after the first one.
- **Token Generation Time:** The total time spent continuously generating tokens during the decode phase (Number of tokens generated × ITL).

**Overall System Latency**

- **Total Latency / End-to-End Latency (E2EL):** The total time perceived by the user from hitting "send" to getting the final token.
- **Formula:** E2EL = TTFT + (Total_Generated_Tokens - 1) × ITL

### 5.2 Reducing Cache Size (MQA & GQA)

How do we shrink the KV cache? Instead of giving every Attention Head its own K and V, we force multiple Attention Heads to share the same K and V vectors.

These attention mechanisms aim to shrink the KV footprint or reduce memory transfers, preserving model quality while optimizing generation speed:

- **Multi-Query Attention (MQA):** All heads in a layer share a single set of K and V instead of maintaining their own. This significantly reduces cache size and memory reads during decode, though it usually comes at some cost in model quality.
- **Grouped Query Attention (GQA):** A middle ground between MQA and full multi-head attention. Query heads are split into groups, and each group shares one K and V. Retains MQA's efficiency while keeping the accuracy of multi-head attention (Used in Llama 2 & 3, Mistral).
- **Multi-head Latent Attention (MLA):** Stores K and V in a learned low-dimensional latent space and projects them in and out as needed. Reduces KV cache size and bandwidth while trading a small amount of extra compute for the projections. (Famously used in DeepSeek V2/V3).
- **Grouped Tied Attention (GTA):** Ties keys and values within each group, reducing cache size and memory traffic at decode while maintaining GQA-level quality. The tied KV vectors are created using a single projection, cached, and used as the Value. For the Key, only the first half is taken unrotated, while the second half comes from a separate one-head RoPE projection broadcast across groups. Halves the KV cache, cuts memory traffic, and doubles arithmetic intensity compared to GQA.
- **Grouped Latent Attention (GLA):** Stores K and V in a latent representation optimized for efficient parallel sharding. Achieves MLA-like compression but is more hardware-friendly and suitable for distributed inference.

### 5.3 Handling Fragmentation (PagedAttention)

As sequences generate, KV cache grows dynamically, causing memory fragmentation (wasted GPU memory).

**PagedAttention** (used in vLLM) solves this by splitting the KV cache into fixed-size "blocks" or "pages" that don't need to be stored contiguously in memory, similar to an OS virtual memory system. This allows for massively larger batch sizes during serving.

#### 1. The Pre-PagedAttention Problem: Why only 30-40% utilization?

Before PagedAttention (e.g., in systems like FasterTransformer), the KV cache was stored in contiguous (unbroken) chunks of GPU memory. This caused massive memory waste due to Fragmentation.

- The "Unknown Length" Problem: When a user sends a prompt, you don't know if the LLM will reply with 10 tokens or 1,000 tokens.
- Internal Fragmentation (The biggest culprit): Because memory had to be contiguous, the system had to guess and pre-allocate memory for the maximum possible sequence length (e.g., 2,048 tokens). If the LLM only generated 20 tokens, the remaining 2,028 token spaces were locked up and entirely wasted.
- External Fragmentation: As requests finished and freed up memory, the GPU VRAM became a "checkerboard" of used and free contiguous chunks. A new request might need a chunk of 1,000 tokens, but even if you had 1,000 spaces free, you couldn't use them because they were scattered in smaller, disconnected gaps.
- The Result: Up to 60-70% of the GPU's KV cache memory was allocated but entirely empty.

#### 2. The Solution: PagedAttention (The OS Analogy)

PagedAttention borrows a classic concept from Operating Systems: Virtual Memory with Paging.

- Instead of forcing the KV cache to be one giant contiguous block, we chop it up into small, fixed-size Blocks (usually 16 tokens per block).
- Logical/Virtual Blocks: As far as the LLM's attention mechanism knows, the tokens are in perfect, continuous sequence (e.g., Token 1 to Token 32).
- Physical Blocks: In the actual GPU VRAM, these blocks can be scattered anywhere. They do not need to be next to each other.
- The Block Table: A mapping system that tells the GPU: "Logical tokens 1-16 are in Physical Block 50. Logical tokens 17-32 are in Physical Block 12."

#### 3. Step-by-Step: How it works during Generation

- Prefill: A user sends a prompt with 30 tokens.
- Allocation: The system allocates exactly two blocks (16 tokens each = 32 token spaces). Waste is practically zero (just 2 empty slots in the last block).
- Decode Step: The model generates tokens. Once the current block hits 16 tokens, the system dynamically allocates exactly one new block anywhere in VRAM and updates the Block Table.
- Attention Calculation: When calculating Attention, the GPU looks at the Block Table, fetches the scattered physical blocks, and computes the math as if they were contiguous.

#### 4. How does this improve "Speed"? (Crucial Interview Distinction)

- Interview Trap: PagedAttention does not make the actual matrix multiplication math faster. It makes the server throughput faster.
- Near 100% Utilization: Because we eliminated fragmentation, we reclaimed that 60% of wasted memory.
- Massive Batch Sizes: Because we have so much more free memory, we can fit way more users (concurrent requests) into the same GPU.
- The Speedup: Going from a batch size of 8 to a batch size of 32 means the server is generating 4x as many tokens per second (Throughput). vLLM achieved a 2x-4x throughput improvement over older systems purely through this memory trick.


#### Point 1: Memory Sharing (Copy-on-Write)

Because PagedAttention uses Block Tables, multiple sequences can share the same physical blocks!

Example: If you use Beam Search (where the model explores 3 different possible sentences from a single prompt), the initial prompt's KV cache is just stored once in physical memory. All 3 beams' Block Tables simply point to that same physical block. If Beam 2 changes a token, it creates a copy of just that specific block (Copy-on-Write). This saves massive memory.

#### Point 2: It Enables Continuous Batching

Continuous Batching (swapping requests in and out the exact millisecond they finish) is almost impossible with contiguous memory because you'd have to constantly defragment the GPU VRAM. PagedAttention's dynamic block allocation is the engine that makes Continuous Batching possible.

#### Point 3: Block Size Trade-offs

"Why 16 tokens per block? Why not 1 or 256?"

- Too large (e.g., 256): You get Internal Fragmentation again. If a generation stops at 257 tokens, you waste 255 spaces in the second block.
- Too small (e.g., 1): The Block Table becomes massive, and the GPU wastes too much time doing memory lookups instead of math. 16 to 32 is the sweet spot.

### 5.4 Practitioner: vLLM Hyperparameters When Deploying a Model

When deploying a model on vLLM (KV cache, PagedAttention, continuous batching), tune these hyperparameters:

| Hyperparameter | What to try | Why it matters |
| --- | --- | --- |
| **--max-model-len** | Match your use case (e.g. 4k, 8k, 32k, 128k). Start lower if OOM. | Caps max sequence length and reserves KV cache memory. Higher = more VRAM for cache, fewer concurrent requests. |
| **--gpu-memory-utilization** | Default 0.90. Lower to 0.85–0.88 if OOM; can push 0.92–0.95 if stable. | Fraction of VRAM used for KV cache + model. Too high → OOM on long context or large batches. |
| **--block-size** | 16 or 32 (default 16). Align with hardware (e.g. 128-bit). | PagedAttention block size. Larger blocks = less block-table overhead, more internal fragmentation. |
| **--dtype** | bfloat16 (safest), float16, or auto. FP8 on H100/B200 for throughput. | KV cache and compute precision. BF16 avoids overflow; FP8 reduces cache size and bandwidth (decode-bound). |
| **--max-num-seqs** | 1–256+. Increase for higher throughput; decrease if OOM or high latency. | Max concurrent sequences in a batch. Drives continuous batching and KV cache usage. |
| **--enable-prefix-caching** | True when you have repeated system/prompt prefixes. | Reuses KV cache for shared prefix → lower TTFT and prefill cost. |
| **--enforce-eager** | False (default); True to disable CUDA graphs for debugging. | When True, no CUDA graph capture (slower but easier to debug). |
| **--tensor-parallel-size** | 1 (single GPU), 2, 4, 8 for multi-GPU. | Shards model and KV cache across GPUs for large models or longer context. |

**Quick checklist:** If you hit OOM, try lowering `--gpu-memory-utilization` or `--max-model-len`, or enable `--enable-chunked-prefill` (splits prefill into chunks). For higher throughput, increase `--max-num-seqs` and ensure Flash Attention (or equivalent) is enabled.

---

## 6. Next Things

### 1. Continuous Batching (aka In-Flight Batching)

- **The Problem:** In a real server, User A asks a short question (10 tokens) and User B asks a long question (100 tokens). If you batch them together statically, the GPU wastes time doing nothing for User A while waiting for User B to finish.
- **The Solution:** Continuous Batching. Instead of waiting for the whole batch to finish, the system ejects User A's request the exact millisecond it's done and instantly inserts User C's new prompt into the batch.
- **Interview Drop:** "Continuous batching operates at the iteration level, not the request level. It is the primary reason PagedAttention (vLLM) is necessary, because we need to dynamically allocate KV cache memory on the fly as requests enter and exit."

### 2. KV Cache Quantization

- **The Problem:** Look back at your memory formula. The only variable we can easily shrink without changing the model architecture is Bytes_Per_Param.
- **The Solution:** Instead of storing the KV cache in FP16 (2 bytes per number), we compress it down to INT8 (1 byte) or even INT4 (0.5 bytes).
- **Interview Drop:** "Because the decode phase is Memory-Bandwidth Bound, fetching less data makes generation much faster. KV Cache Quantization trades a tiny bit of model accuracy for a massive boost in generation speed and server capacity."

### 3. Prefix Caching (Prompt Caching)

- **The Problem:** Imagine a company using an LLM. Every single user request starts with a massive hidden system prompt: "You are a helpful assistant. Here are the company rules:[1,000 words]..." Recomputing the KV cache for those same 1,000 words for every user wastes massive compute.
- **The Solution:** Radix Trees / Prefix Caching. You compute the KV cache for the system prompt exactly once, keep it permanently in GPU memory, and let all new user requests just point to that existing cache.
- **Interview Drop:** "Prefix caching drastically reduces the Prefill compute time and lowers the Time-To-First-Token (TTFT) for users."

### 4. Eviction Policies (What happens when we run out of memory?)

- **The Problem:** If a user talks to a chatbot for 3 hours, the KV cache grows to infinite size and the GPU runs out of memory (OOM).
- **The Solution:** We have to start deleting (evicting) old memory. But how?
- **Sliding Window Attention (SWA):** Used by Mistral. Only keep the last \(W\) tokens in the cache (e.g., last 4,096 tokens). Drop the rest.
- **StreamingLLM / Attention Sinks:** A fascinating discovery. If you drop the very first token of a prompt, the model goes crazy. But if you keep the first ~4 tokens (Attention Sinks) permanently in the KV cache, plus a sliding window of recent tokens, the model can generate text infinitely without crashing.

### 5. Positional Encodings (RoPE) & The Cache

- **The Problem:** If all tokens are just stored in a big matrix, how does the model remember the order of the words? "The dog bit the man" is different from "The man bit the dog".
- **The Solution:** Rotary Position Embeddings (RoPE).
- **Interview Nuance:** You must apply positional encodings to the Query (Q) and Key (K) vectors BEFORE you save K to the KV cache. Once K is in the cache, its positional information is baked into the numbers permanently.



Reference:
1. [KV Cache Blog](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/)