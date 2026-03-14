# GPU Inference

## 1. The GPU Landscape: Why Hardware Matters

### Memory: Where Data Lives

- **HBM (The Warehouse):** Where the billions of model weights live. High capacity, but slow access.
- **SRAM (The Workbench):** Where the math actually happens. Tiny capacity, but blazing fast.
- **The Problem:** During inference, we spend most of our time moving weights from the Warehouse to the Workbench.

### Compute: Tensor Cores vs. CUDA Cores

When the data finally reaches the "Workbench" (SRAM), two types of engines can process it:

- **CUDA Cores (The Generalists):**
  - **Role:** Handle standard, element-wise math (additions, Softmax, LayerNorm, ReLU).
  - **Efficiency:** They process one operation at a time.
  - **Analog:** A skilled carpenter with a manual saw.

- **Tensor Cores (The Matrix Specialists):**
  - **Role:** Designed specifically for Deep Learning. They perform massive Matrix-Matrix Multiplications (GEMM) in a single clock cycle (e.g., \(4 \times 4\) or \(16 \times 16\) matrices).
  - **Efficiency:** Can be 10x–20x faster than CUDA cores for matrix math.
  - **Analog:** A giant industrial press that stamps out a whole shape at once.

**Note:** Flash Attention 2 was a major upgrade because it reorganized math to stay on Tensor Cores for as long as possible (80%+ utilization), whereas older kernels kept dropping back to the slower CUDA cores for intermediate steps.

---

## 2. Deep Dive: The "Arithmetic Intensity = 1" Proof

"Why is the Decode phase (generating tokens) slower/less efficient than the Prefill phase (processing the prompt)?"

**The Math of generating one token (Batch Size = 1):**

Imagine a single Linear Layer in a transformer (e.g., Llama 3 8B) where the hidden dimension \(d = 4096\). To generate the next token, we must multiply a Weight Matrix (\(W\)) by an Input Vector (\(x\)).

- **Weight Matrix (\(W\)):** Shape \((d \times d)\). Size \(= d^2\).
- **Input Vector (\(x\)):** Shape \((d \times 1)\). Size \(= d\).

#### 1. Calculate FLOPs (The Math)

To compute \(y = Wx\), for every row in \(W\), we do a dot-product with \(x\).

- 1 dot-product of length \(d\) ≈ \(d\) multiplications + \(d\) additions = \(2d\) operations.
- There are \(d\) rows in \(W\).
- **Total FLOPs** \(= d \times 2d = 2d^2\).

#### 2. Calculate Bytes (The Memory)

To do this math, we must load every parameter of \(W\) from HBM into SRAM.

- Number of parameters \(= d^2\).
- In FP16/BF16, each parameter is 2 bytes.
- **Total Bytes loaded** \(= 2d^2\).

#### 3. Calculate Arithmetic Intensity (AI)

\[
AI = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2d^2}{2d^2} = 1
\]

**The Verdict:** Modern GPUs like the A100 have a "Balance Point" of roughly 156. Since our AI is 1, we are using <1% of the GPU's computational power. The GPU is essentially "starving" for data.

---

## 3. How vLLM "Pushes the Roofline"

vLLM's goal is to turn that AI = 1 into AI = 100+. It does this by using Batching.

**The Math of Batching (Batch Size = \(B\)):**

If we process \(B\) requests at the same time, \(x\) becomes a matrix \(X\) of shape \((d \times B)\).

- **FLOPs:** \(2 \times d \times d \times B = 2Bd^2\)
- **Bytes:** We load the weight matrix \(W\) once and reuse it for all \(B\) requests. Total Bytes \(= 2d^2\).
- **New AI:** \(\frac{2Bd^2}{2d^2} = B\).

By increasing the Batch Size (\(B\)), we directly increase the Arithmetic Intensity. If \(B = 128\), our AI is 128, which gets us very close to the GPU's "Compute Roof" (156).

**vLLM's Two Secret Weapons:**

- **Continuous Batching:**
  - **The Old Way (Static):** If 3 requests are in a batch and one finishes early, the GPU sits idle waiting for the other two.
  - **The vLLM Way:** As soon as one request finishes, vLLM inserts a new request into the batch immediately. This keeps the Batch Size (\(B\)) consistently high, keeping the AI high.

- **PagedAttention:**
  - The biggest blocker to high batch sizes is the KV Cache (it eats up VRAM).
  - PagedAttention manages KV cache memory like an OS manages virtual memory (using "pages"). This eliminates memory fragmentation and reduces waste by up to 95%.
  - **Result:** Because memory is used so efficiently, you can fit much larger batches (\(B\)) into the same GPU. Larger \(B\) = Higher AI = Better GPU Utilization.

---

## 4. Practitioner's Debugging Guide

### What the Metrics Tell You

| Metric | Observation | Meaning |
| --- | --- | --- |
| Throughput vs. Batch Size | Throughput increases linearly as you increase Batch Size. | You are currently Memory-Bound (this is good; keep increasing batch size). |
| Throughput Plateaus | Increasing Batch Size no longer increases tokens/sec. | You have hit the Compute-Bound "Roof." Adding more batching won't help speed. |
| Latency per Token | Latency jumps suddenly as Batch Size increases. | You are likely thrashing the KV cache or hitting PCIe/Interconnect bottlenecks. |

### What to Check in vLLM

- **vllm:num_requests_running:** If this is low, your system isn't batching enough. You aren't "pushing the roofline."
- **vllm:gpu_cache_usage_perc:** If this is 100% but your throughput is low, your KV cache is too small to allow for large, efficient batches.

### What to Monitor & Tune

- **Monitor TPOT (Time Per Output Token):**
  - If TPOT is high but stays the same when you double the batch size → You are Memory-Bound. (Good! Increase batching).
  - If TPOT increases significantly when you increase batch size → You have hit the Compute-Bound limit or a PCIe bottleneck.

- **Check Precision (FP16 vs FP8):** On H100s, use FP8. It doubles the Tensor Core throughput and halves the HBM traffic. This is the fastest way to "cheat" the memory wall.

- **Use quantization:** If you are strictly memory-bound, switching to FP8 or 4-bit (AWQ) cuts the "Bytes" in the \(AI = \text{FLOPs}/\text{Bytes}\) equation in half (or more), effectively doubling your throughput.

- **KV Cache Tuning:** In vLLM, use `--gpu-memory-utilization`. If you have long context lengths (e.g., 32k+), Flash Attention is mandatory, or you will OOM immediately because the \(N^2\) matrix will exceed HBM capacity.

---

## 5. Final Summary & Common Gotchas

### Recap

- **Standard Inference:** \(AI = 1\). The GPU is idle 99% of the time, waiting for weights from HBM.
- **vLLM Strategy:** Use PagedAttention to free up memory, which allows for massive Continuous Batching.
- **The Result:** By processing many users at once, the cost of loading the weights from HBM is "amortized" across all users, pushing the Arithmetic Intensity from 1 toward 150+, finally using the full power of the GPU cores.

### Q&A

- **Q: Why do we need Flash Attention if we have high-bandwidth memory (HBM3)?**  
  **A:** Because compute speed (Tensor Cores) is growing 10x faster than memory bandwidth. Even with HBM3, the "gap" is getting wider. Flash Attention minimizes the need to use that bandwidth by keeping data on-chip.

- **Q: Why is Batch Size the most important lever for throughput?**  
  **A:** Because it increases the Arithmetic Intensity. It allows us to load the model weights once and perform math for many users, moving us from the "Memory-Bound" slope of the Roofline to the "Compute-Bound" plateau.

- **Q: What is the difference between CUDA and Tensor cores in the context of Attention?**  
  **A:** Tensor cores handle the heavy lifting of \(QK^T\) and \(PV\) (Matrix-Matrix). CUDA cores handle the "glue" like Softmax scaling and mask application. Flash Attention 2/3 aims to minimize the "glue" time to keep the "heavy lifting" Tensor Cores running.
