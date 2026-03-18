


Fundamental and Advanced Distributed Training Concepts

## 1. Single GPU Memory Bottlenecks
Memory in deep learning training is split into two categories: **Static** and **Dynamic**.

**Static Memory (The Model & Optimizer)**
Defined by the model architecture. If $\Psi$ is the number of parameters, using the Adam optimizer with **Mixed Precision (BF16/FP32)** requires **16 bytes per parameter ($16\Psi$)**:
*   **Model Weights (BF16):** $2\Psi$ bytes
*   **Gradients (BF16):** $2\Psi$ bytes
*   **Master Weights (FP32):** $4\Psi$ bytes (Needed for stable optimizer updates)
*   **Optimizer States (FP32):** $8\Psi$ bytes (Adam keeps 1st & 2nd moments, $4\Psi$ each)
*   *Note:* A 70B parameter model requires ~1.1 TB of static memory alone—impossible for a single 80GB A100 GPU [1]. 

**Dynamic Memory (Activations)**
Activations must be stored during the forward pass to compute gradients in the backward pass.
*   Grows **linearly** with Batch Size.
*   Grows **quadratically** with Sequence Length (due to the attention matrix in Transformers) [1].

---

## 2. Taming Dynamic Memory
When activations cause Out of Memory (OOM) errors, we use two main techniques:

*   **Activation Recomputation (Gradient Checkpointing):** 
    *   Instead of storing all intermediate activations, save only specific "checkpoints". Recompute the missing ones during the backward pass.
    *   *Trade-off:* Saves massive memory but increases compute time (up to 30-40% slower) [1].
    *   *Pro-Tip:* Use **Selective Checkpointing**—only drop the memory-heavy Multi-Head Attention (MHA) activations and keep the MLP ones. Saves ~70% memory for only ~2.7% extra compute [1].
*   **Gradient Accumulation:** 
    *   Split a large batch into smaller "micro-batches". Run forward/backward passes sequentially, accumulating gradients without updating weights, then do one global `optimizer.step()`. 
    *   *Trade-off:* Reduces peak dynamic memory but increases step time [1].

---

## 3. Data Parallelism (DP) & Communication Primitives
If the model fits on a single GPU but training is too slow, we scale horizontally using DP.
*   **How it works:** Replicate the *entire* model, gradients, and optimizer states on $N$ GPUs. Each GPU processes a different micro-batch simultaneously (SIMD) [1].
*   **Communication:** Uses the **All-Reduce** primitive to sum the gradients from all GPUs so every GPU has the same global gradient before the optimizer step [1].
*   **Optimization:** Overlap communication and computation. Use PyTorch hooks (e.g., `post_accumulate_grad_hook`) to trigger All-Reduce for layer $L_2$'s gradients while layer $L_1$'s gradients are still being computed [1].

---

## 4. Scaling Static Memory: ZeRO & FSDP
When the model is too big for a single GPU, simple DP fails. We use **ZeRO (Zero Redundancy Optimizer)** to shard (partition) the $16\Psi$ static memory across $N_d$ GPUs [1]. 

| Strategy | What is Sharded? | Memory Formula per GPU | Communication Primitives |
| :--- | :--- | :--- | :--- |
| **Vanilla DP** | None | $16\Psi$ | **All-Reduce** (Gradients) |
| **ZeRO-1** | Opt. States & Master Weights | $4\Psi + \frac{12\Psi}{N_d}$ | **All-Reduce** (Grads) $\rightarrow$ **All-Gather** (Updated Weights) |
| **ZeRO-2** | Opt. States, Master Wts, Gradients | $2\Psi + \frac{14\Psi}{N_d}$ | **Reduce-Scatter** (Grads) $\rightarrow$ **All-Gather** (Updated Weights) |
| **ZeRO-3 (FSDP)** | Everything (Params, Grads, Opt States) | $\frac{16\Psi}{N_d}$ | **All-Gather** (Fwd Params), **All-Gather** (Bwd Params), **Reduce-Scatter** (Bwd Grads) |

*   *Communication Tax of ZeRO-3:* Total comm cost per iteration is $3\Psi$. Handled efficiently by **prefetching** the next layer's parameters to overlap network IO with GPU compute [1].
*   *Note:* **FSDP (Fully Sharded Data Parallel)** is PyTorch's official native implementation of the ZeRO-3 concept [1].

---

## 5. Distributed Orchestration with Ray
*   **Ray Core Primitives:** 
    *   **Tasks (`@ray.remote`):** For stateless, parallel functions (e.g., mapping over data) [1].
    *   **Actors (`@ray.remote` classes):** For stateful, persistent microservices (e.g., maintaining a global counter or serving a model) [1].
*   **Why Ray Train over PyTorch DDP?** PyTorch DDP is great for single-node. Ray Train abstracts away multi-node cluster orchestration, node failure recoveries (fault tolerance), unified CPU-GPU data streaming, and elastic scaling [1]. To use FSDP in Ray, you simply wrap your model with `ray.train.torch.prepare_model(model, parallel_strategy="fsdp")`[1].




Yes! While your cheat sheet perfectly covers **Data Parallelism (DP) and ZeRO/FSDP**, a Staff ML Engineer is expected to zoom out and look at the entire **system architecture, hardware topologies, and alternative parallelism strategies**. 

When scaling to massive models (e.g., GPT-3, LLaMA-3 70B+) or huge clusters, FSDP alone isn't enough. Here are the crucial missing concepts you should add to your preparation:


### 1. 3D Parallelism (Beyond Data Parallelism)
FSDP is a form of Data Parallelism. For massive models
*   **Tensor Parallelism (TP) [Intra-layer]:** Instead of sharding layers sequentially, TP slices the actual matrix multiplications *inside* a layer (like an MLP or Attention block) across multiple GPUs. 
    *   *Used in:* Megatron-LM.
    *   *Trade-off:* Requires incredibly high communication bandwidth because GPUs must All-Reduce *during* the forward and backward passes of every single layer.
*   **Pipeline Parallelism (PP) [Inter-layer]:** Slicing the model vertically. GPU 0 gets layers 1-10, GPU 1 gets layers 11-20, etc.
    *   *The "Bubble" Problem:* Sequential execution means GPU 1 sits idle waiting for GPU 0 to finish. Solved by injecting "micro-batches" into the pipeline, but a small idle time (the "pipeline bubble") always remains.

### 2. Context/Sequence Parallelism (The 4th Dimension)
With the rise of massive LLM context windows (e.g., 100K+ tokens), dynamic memory blows up quadratically even with a batch size of 1.
*   **Ring Attention / DeepSpeed Ulysses:** Shards the *sequence length* across multiple GPUs. GPUs pass attention key-value blocks around in a ring topology to compute global attention without any single GPU holding the full sequence in memory.

### 3. Hardware Interconnect Topology (Hardware-Software Co-design)
Staff MLEs must know how to map parallel strategies to physical networking hardware:
*   **NVLink / NVSwitch:** Ultra-fast intra-node communication (between the 8 GPUs inside a single server). Bandwidth is ~400-900 GB/s.
*   **InfiniBand (IB) / RoCE:** Inter-node communication (between different servers in the cluster). Bandwidth is much slower, ~50-400 Gbps (Gigabits, not Bytes).
*   **The Golden Rule of 3D Parallelism:** Because **Tensor Parallelism (TP)** requires massive communication, you *must* restrict TP to a single node (using NVLink). You use **Pipeline Parallelism (PP)** and **Data Parallelism (FSDP/ZeRO)** across different nodes because they send fewer, larger chunks of data and can tolerate slower InfiniBand.

### 4. FlashAttention (Kernel / IO Awareness)
A Staff MLE should understand *why* attention is slow. It's usually not compute-bound; it's **Memory-Bandwidth Bound**.
*   Standard Attention reads/writes the large $N \times N$ attention matrix to HBM (High Bandwidth Memory - the GPU's main RAM) multiple times.
*   **FlashAttention:** Uses tiling to compute the exact attention output while keeping data in the ultra-fast, tiny SRAM (L1 Cache) on the GPU chip, drastically reducing HBM read/writes. It reduces memory usage from $O(N^2)$ to $O(N)$ and speeds up wall-clock time.

### 5. Profiling Metrics (MFU & HBU)
How do you know if your distributed setup is good?
*   **MFU (Model FLOPs Utilization):** The ratio of observed throughput (FLOPs) to the theoretical maximum peak FLOPs of the GPU. A great distributed setup achieves 40-60% MFU. If it's 10%, your GPUs are starved for data or choked by network communication.
*   **Stragglers:** In synchronous training (like All-Reduce), the entire cluster moves only as fast as the *slowest* GPU. Identifying degraded hardware (a slightly overheating GPU) is a massive part of distributed training at scale.


## 🚨 Gotchas

1.  **The Mixed Precision Illusion:** A common misconception is that switching from FP32 to BF16 cuts your total static memory in half. **It does not.** You must maintain an FP32 copy of the Master Weights and Optimizer States for numerical stability. Total static memory remains strictly at $16\Psi$ [1]. The true benefit of mixed precision is faster compute (TFLOPS) and halving the *dynamic* activation memory [1].
2.  **ZeRO-1 vs. Master Weights:** When people say ZeRO-1 "shards optimizer states", they implicitly mean it shards the optimizer states **AND** the FP32 Master Weights [1]. If you only sharded Adam states, the $4\Psi$ master weights would still bottleneck you.
3.  **Communication Overhead vs. Batch Size:** If your micro-batch size is too small in a distributed setting, GPU computation finishes too quickly, and the GPU will idle while waiting for All-Reduce/All-Gather over the network. Network bandwidth becomes the bottleneck.
4.  **ZeRO-3 is not always the answer:** If your model fits into ZeRO-2 memory, use ZeRO-2. ZeRO-3 imposes a $3\Psi$ communication cost per iteration compared to ZeRO-2's $2\Psi$ cost. Aggressive sharding is a trade-off against network bandwidth [1].

---

## 🧠 Questions and Answers

**Q1: "We just switched our LLM training script to use BF16 mixed precision. Why didn't our static memory footprint drop at all?"**
**A:** Because mixed precision training requires maintaining FP32 "Master Weights" ($4\Psi$) and FP32 Adam Optimizer states ($8\Psi$) to preserve update stability and avoid precision loss. Thus, static memory stays exactly at 16 bytes per parameter. It only reduces the memory size of forward-pass *activations* and speeds up matrix multiplications [1].

**Q2: "We ran out of memory, so we turned on Full Gradient Checkpointing (Activation Recomputation). Memory is fine now, but our training time increased by 40%. How can we speed this up without OOMing?"**
**A:** Implement **Selective Checkpointing**. Instead of throwing away and recomputing *all* intermediate activations, we only discard the memory-heavy Multi-Head Attention (MHA) activations—which scale quadratically with sequence length—while keeping the lightweight MLP layer activations in memory. This usually recovers almost all the memory savings for a fractional (~2-3%) compute penalty [1].

**Q3: "If ZeRO-3 (FSDP) shards all the parameters across the cluster, how does a GPU perform the forward pass without running out of memory when evaluating a 70B parameter model?"**
**A:** ZeRO-3/FSDP operates layer-by-layer (or block-by-block). During the forward pass, it issues an **All-Gather** to temporarily reconstruct the full weights for *only that specific layer*. It computes the forward pass for that layer, and then immediately deletes/flushes those weights from memory before moving to the next layer. Prefetching is used to gather layer $N+1$ while computing layer $N$ to hide the communication latency [1].

**Q4: "In ZeRO-1, we perform an All-Reduce on gradients. In ZeRO-2, we perform a Reduce-Scatter. What is the fundamental difference, and why does ZeRO-2 save more memory?"**
**A:** In ZeRO-1's All-Reduce, every GPU receives the *entire* summed global gradient, updates the weights locally, and then discards the gradient parts it doesn't own. 
In ZeRO-2, **Reduce-Scatter** performs the sum and simultaneously partitions the result, so each GPU *only ever receives the shard of the gradient it is responsible for*. This prevents the GPU from having to materialize the entire gradient tensor in memory at once, saving $2\Psi \times (1 - \frac{1}{N_d})$ memory per GPU [1].

**Q5: "We are training a massive model using FSDP across 64 nodes (512 GPUs). The interconnect between nodes is heavily congested, slowing down training. How can we re-architect our parallelism to reduce inter-node network traffic?"**
**A:** FSDP (ZeRO-3) forces massive inter-node communication because it requires an All-Gather and Reduce-Scatter for *every layer* across the entire cluster. We should switch to **Hybrid Parallelism (HSDP)** or **3D Parallelism**. We can group FSDP to operate *only within a node* (or a small pod) where NVLink is fast, and use standard Data Parallelism or Pipeline Parallelism *across* the nodes, as they have much lower communication frequency over the slower InfiniBand network.

**Q6: "Your team implements Pipeline Parallelism and notices GPU utilization is dropping significantly. What is causing this, and how do you fix it?"**
**A:** This is the **Pipeline Bubble**. Because layers are split across GPUs, downstream GPUs are starved waiting for upstream GPUs to pass the activations forward. To fix this, we implement **Micro-batching (e.g., 1F1B scheduling)**. We split the batch into tiny micro-batches and feed them into the pipeline sequentially, so while GPU 4 is processing micro-batch 1, GPU 1 is already processing micro-batch 4. This shrinks, but does not entirely eliminate, the bubble.

**Q7: "Why do we say Standard Attention is 'Memory-Bandwidth Bound' while large Linear layers (MLPs) are 'Compute Bound'?"**
**A:** In MLPs, we do a massive amount of math (matrix multiplication) relative to the amount of weights we load from GPU memory. The GPU's math cores (Tensor Cores) are the bottleneck. In standard Attention, creating the $N \times N$ attention matrix requires relatively simple math, but reading and writing that massive matrix to the GPU's HBM takes longer than the math itself. The GPU math cores sit idle waiting for memory. FlashAttention solves this by fusing the operations into SRAM.

##  Deep Dive: Communication Primitives in FWD / BWD Passes

To understand ZeRO and FSDP, you must understand exactly *when* and *why* GPUs talk to each other over the network (usually via NVLink/NCCL). 

### 1. All-Gather (Reconstructing Sharded Data)
**What it does:** Every GPU broadcasts its piece of a tensor to all other GPUs. At the end, *every GPU has the complete, concatenated tensor*.
**When is it used?** 
*   **ZeRO-3 / FSDP (Forward Pass):** Because parameters are sharded across the cluster, no GPU has the full weights for Layer $L$. Right before Layer $L$ executes, an **All-Gather** is called to collect the missing weight shards from all other GPUs. The forward pass is computed, and the gathered weights are immediately discarded to free up memory.
*   **ZeRO-3 / FSDP (Backward Pass):** The exact same process happens. To compute the gradients for Layer $L$, the GPU must temporarily reconstruct the full weights using **All-Gather**. 

### 2. Reduce-Scatter (Summing and Sharding Simultaneously)
**What it does:** It performs a reduction (usually summing) across all GPUs, but instead of giving the full result to everyone, it slices the result into chunks and gives **one specific chunk to each GPU**.
**When is it used?** 
*   **ZeRO-2 & ZeRO-3 (Backward Pass):** After the backward pass computes the local gradients for Layer $L$ on each GPU's micro-batch, we need to sum these gradients globally. Instead of materializing the huge global gradient tensor on every GPU, **Reduce-Scatter** sums the gradients over the network and directly hands GPU 0 its specific shard of the gradient, GPU 1 its shard, etc. The GPU only holds the gradients for the weights it is responsible for updating.

### 3. All-Reduce (The Vanilla DP Workhorse)
**What it does:** The combination of *Reduce-Scatter* followed by *All-Gather*. It sums the arrays across all GPUs and ensures *every GPU receives the complete, identical summed result*.
**When is it used?** 
*   **Vanilla DP & ZeRO-1 (Backward Pass):** Once the backward pass reaches the input layer, every GPU has a full set of local gradients based on its specific micro-batch. An **All-Reduce** is triggered to sum up the gradients across all GPUs so that every GPU holds the exact same global gradient tensor. 
*   *Note:* This is heavily memory-inefficient for large models because every GPU must have enough RAM to store the entire gradient tensor, which is why ZeRO-2/3 replaces it with Reduce-Scatter.

---

### 💡 The FSDP (ZeRO-3) Lifecycle Summary:

1.  **FWD Pass starts:** Trigger **All-Gather** to reconstruct weights $\rightarrow$ Compute FWD $\rightarrow$ Discard weights.
2.  **BWD Pass starts:** Trigger **All-Gather** to reconstruct weights $\rightarrow$ Compute local gradients $\rightarrow$ Discard weights.
3.  **BWD Pass finishes layer:** Trigger **Reduce-Scatter** on local gradients $\rightarrow$ GPU gets its summed gradient shard $\rightarrow$ GPU updates its shard of the optimizer states and master weights. 

PyTorch FSDP overlaps the All-Gather for Layer $L-1$ over the network while the GPU is busy computing the Backward Pass for Layer $L$. This is called **Compute/Communication Overlap** and is critical for high GPU FLOPS utilization).*