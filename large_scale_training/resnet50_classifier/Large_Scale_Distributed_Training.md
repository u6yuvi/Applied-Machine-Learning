# Training guide: large-scale image classification

Everything in this guide is grounded in the actual code in this repo — `lightning_main.py`, `model_resnet50.py`, `bce_loss.py`, `augmentation.py`. Where the code does something for a reason, the reason is here. Where the code has a sharp edge, that is in the gotchas.

---

## 1. Model: ResNet-50 v1.5

The backbone is a custom `ResNet50` (`model_resnet50.py`), not a torchvision pretrained model. Three things about it matter for training behavior:

### Stride placement (v1.5 vs v1)

In v1, the **first** 1×1 conv in each bottleneck carries the stride-2 downsample. In v1.5, **the 3×3 conv** does. This is a small architectural choice from Microsoft's recipe that gives roughly **+0.5% top-1** at no extra cost. It matters because if you compare your numbers against v1 baselines, there is already a built-in gap.

### Kaiming He initialization

All conv layers use `kaiming_normal_(mode="fan_out", nonlinearity="relu")`. BN layers are initialized with `weight=1`, `bias=0`. This is the standard setup that keeps variance stable through deep ReLU networks.

### Zero-gamma on residual BN

After the general init, the code walks every `Bottleneck` block and sets `bn3.weight = 0` — the **last** BN in the residual branch. The effect: at initialization, each residual block contributes **nothing** to the output (the residual path multiplies by zero), so the network behaves like a shallower model early in training and deepens gradually as BN gamma grows. This is a well-known trick from the original ResNet paper's appendix and Facebook's large-batch training work. It stabilizes early training, especially at high learning rates.

The code also zeros the BN gamma inside **downsample** paths. This is more aggressive than some recipes — worth knowing if you ever see divergence on the first epoch.

---

## 2. Loss functions

### Cross-entropy (`--loss_type cross_entropy`)

Standard `nn.CrossEntropyLoss` on integer class indices. Softmax is implicit. This is the default and the one most people should start with. Label smoothing via the CLI flag is **silently zeroed out** for this path (see gotchas).

### BCE with logits (`--loss_type bce_with_logits`)

`ImageNetBCEWithLogitsLoss` in `bce_loss.py`. Each class logit goes through **sigmoid independently** (not softmax), and binary cross-entropy is computed against a **one-hot** (or label-smoothed one-hot) target vector.

**Why the custom class instead of `nn.BCEWithLogitsLoss`?** The stock PyTorch version with `reduction="mean"` averages over **both** the batch dimension **and** the class dimension. For ImageNet with 1000 classes, that shrinks the loss by ~1000× compared to cross-entropy, making gradients tiny and training effectively dead at normal learning rates. This repo uses **`reduction="sum"` divided by batch size only**, which keeps the loss magnitude in the same ballpark as cross-entropy. This is a real foot-gun that is easy to miss if you wire up BCE naively.

Label smoothing is supported here: the one-hot target is softened so mass is spread to non-target classes, reducing overconfident predictions.

### BCE on probabilities (`--loss_type bce`)

Same idea, but applies `sigmoid` first and feeds probabilities into `binary_cross_entropy`. Numerically less stable than the logits variant, especially under **mixed precision** where small probabilities can underflow. Prefer `bce_with_logits` unless you have a specific reason.

### Bias initialization for BCE (`--init_bce_bias`)

Sets the final FC layer bias to **\(-\log(\text{num\_classes})\)**, which makes initial sigmoid outputs roughly **1/C** (uniform). Without this, initial outputs are ~0.5 per class (sigmoid of zero), and the network starts training from a very different loss surface than cross-entropy would. This initialization closes most of the accuracy gap that people see when switching from CE to BCE.

---

## 3. Data pipeline

### ImageNet transforms (224×224)

**Training:** `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip(0.5)` → `ColorJitter(0.2, 0.2, 0.2, 0.1)` → `RandAugment(num_ops=2, magnitude=9)` → `ToTensor` → `Normalize(ImageNet mean/std)` → optionally `RandomErasing(p)`.

**Validation:** `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`.

### TinyImageNet transforms (64×64)

Same chain, but `Resize(72)` → `RandomCrop(64)` (train) / `CenterCrop(64)` (val). The augmentation strengths are the same, which may be too aggressive for the smaller resolution — something to watch if val accuracy plateaus early.

### DataLoader settings

`shuffle=True` (train), `pin_memory=True`, configurable `num_workers`. Under DDP, Lightning replaces the sampler with a `DistributedSampler` automatically so each rank sees a disjoint shard. The user-facing `shuffle=True` is still respected within each rank's partition.

---

## 4. Augmentation: Mixup and CutMix

The `MixupCutmixCallback` (`augmentation.py`) runs in `on_train_batch_start`. For each batch:

1. Roll a random number against `--cutmix_prob`.
2. If CutMix: sample λ from Beta(α, α), cut a rectangular patch from a shuffled copy of the batch, paste it in. λ is **recomputed** to match the actual pixel ratio of the pasted region. CutMix **mutates the input tensor in place**.
3. If Mixup: sample λ from Beta(α, α), linearly blend images. Mixup creates a **new** tensor.
4. Store `(y_a, y_b, lam)` on the module as `_current_batch` for mixed-label loss calculation.

**Current state:** The mixed-label loss branch in `training_step` is **commented out**. The module computes loss against the original hard labels. This means:
- With **CutMix**, the input **is** mixed (in place), but the loss uses the **wrong labels** — training signal is incorrect.
- With **Mixup**, the input tensor returned by the callback may not propagate back to `training_step` depending on how Lightning handles the batch reference, so you might be training on **unmixed images** with the right labels.

Bottom line: if you enable `--mixup_alpha` or `--cutmix_alpha`, verify the loss branch is uncommented and working, or you are burning compute.

---

## 5. Optimizer and learning rate schedule

### SGD with two parameter groups

```
Group 1: conv weights + FC weights  → weight_decay=1e-4
Group 2: biases + BN (weight, bias) → weight_decay=0.0
Both:    lr=learning_rate, momentum=0.9
```

Weight decay on BN parameters is a common mistake. BN's affine parameters (γ, β) control scale and shift — regularizing them toward zero fights the network's ability to normalize, which can slow convergence or hurt accuracy.

### Warmup + cosine annealing

When `--warmup_epochs > 0`, a `LambdaLR` scheduler linearly ramps LR from `warmup_start_lr` to `learning_rate` over `warmup_epochs`, then cosine-decays toward `eta_min`. The lambda normalizes by `learning_rate` since PyTorch multiplies the lambda's return value by the base LR.

When warmup is 0, a standard `CosineAnnealingLR` is used directly.

**Why warmup matters at scale:** With large global batches, the initial gradient estimates are noisy relative to the step size. Starting at a small LR and ramping up gives the network a few epochs to find a reasonable basin before taking large steps. This is especially important when the zero-gamma BN init is active — early network dynamics are unusual and full-LR steps can cause instability.

### LR finder

Runs on a **separate, single-GPU Trainer** with `strategy="auto"`. The finder sweeps LR from 1e-6 to 1.0 over one epoch, records loss, and suggests the steepest-descent point. The model's `learning_rate` is updated in place before the main training Trainer is created.

---

## 6. Distributed training (DDP)

### How the script decides

```
GPU count > 1  →  accelerator="gpu", devices=all, strategy="ddp", precision="bf16-mixed"
GPU count == 1 →  accelerator="gpu", devices=1,   strategy="auto", precision="bf16-mixed"
GPU count == 0 →  accelerator="cpu", devices="auto", strategy="auto", precision="32"
```

No manual configuration needed — it auto-detects at startup.

### What DDP actually does per step

1. Each process (one per GPU) draws its **own** mini-batch from its shard of the dataset.
2. Forward pass runs independently on each rank.
3. During backward, gradients are **all-reduced** (averaged) across ranks using NCCL.
4. Every rank applies the **same** optimizer update, so weights stay synchronized.

**Global batch size = per-GPU batch × number of GPUs.** This is not split — each GPU processes the full `--batch_size`. When you go from 1 to 4 GPUs, global batch quadruples. If you do not adjust LR, you are effectively training with a different optimization trajectory.

### Mixed precision (`bf16-mixed`)

On GPU, the Trainer uses **bfloat16 autocast** for eligible ops. BF16 has the same exponent range as FP32 (8 bits) with reduced mantissa, so it rarely produces the inf/NaN issues that FP16 can when loss spikes. No loss scaling is needed with BF16, unlike FP16 AMP.

### `torch.compile`

The ResNet50 backbone is compiled with `mode="reduce-overhead"` inside the Lightning module constructor. This uses CUDA graphs where possible to cut Python/framework overhead per step. Expect the first few iterations to be slow (tracing/compiling), then steady-state throughput improves. If you see recompilations (warnings about graph breaks), the speedup may be partial.

### Gradient clipping

`gradient_clip_val` (default 1.0) performs **global norm clipping** before the optimizer step. In DDP, this happens **after** all-reduce, so all ranks clip consistently. Clipping prevents occasional gradient explosions from destabilizing training — common with mixed precision, aggressive augmentation, or high LR.

### Gradient accumulation

Set to `accumulate_grad_batches=1` (no accumulation). To simulate a larger effective batch without increasing per-GPU memory, you would raise this value. Lightning handles it transparently: backward runs every step, but the optimizer only steps every k batches, and the loss is scaled accordingly. **Effective batch = per-GPU batch × GPUs × accumulation steps.**

---

## 7. Monitoring and logging

### What gets logged

| Metric | Frequency | `sync_dist` |
|--------|-----------|-------------|
| `train_loss`, `train_acc` | Every step | Yes |
| `val_loss`, `val_acc` | Every validation | Yes |
| `learning_rate` | Every step | No (same on all ranks) |
| `grad_norm/total` + per-layer norms | Every 50 steps | Yes (total) |
| `param_mean/*`, `param_std/*`, `param_norm/*` | Every 50 steps | Yes |
| Warmup/cosine LR tracking | Every step | No |

**`sync_dist=True`** means Lightning all-reduces the metric across ranks before logging. Without it, each rank logs its own number and you get inconsistent TensorBoard curves — or worse, only rank 0's value is recorded.

### Gradient and parameter monitoring

Every 50 steps, the module iterates all named parameters and logs L2 gradient norms (for conv, fc, bn layers) and parameter statistics (mean, std, L2 norm). This is useful for catching:
- Gradient explosions (norms spiking) → increase clipping, reduce LR, check data pipeline.
- Gradient vanishing (norms collapsing) → check init, LR, or if BN gamma is stuck at zero.
- Weight norm drift → early sign of instability or that weight decay is too low/high.

### Loggers

Two loggers run in parallel:
- **TensorBoard** — full metric set, good for interactive exploration.
- **Filtered CSV** — only keeps `train_loss`, `val_loss`, `train_acc`, `val_acc`, `learning_rate`, and LR schedule fields. Intentionally strips the per-layer gradient/param metrics to keep CSVs readable.

### Callbacks

- **ModelCheckpoint:** saves top-3 by `val_loss` (min) + always saves `last.ckpt`. Saves at end of validation, not end of training epoch.
- **EarlyStopping:** monitors `val_loss`, patience=10, `strict=False`. The `strict=False` means it will not crash if the monitored metric is missing for an epoch (which can happen during sanity checks or if validation is skipped).
- **LearningRateMonitor:** logs LR per step for TensorBoard visualization.

---

## 8. Checkpointing and resume

Two resume modes:
- `--resume`: finds the **latest** `.ckpt` file (by modification time) in the results checkpoint directory and resumes from it.
- `--resume_from_checkpoint <path>`: resumes from a **specific** checkpoint file.

Lightning checkpoints store model weights, optimizer state, scheduler state, and epoch/step counters. Training continues seamlessly as long as the code matches.

---

## Gotchas

These are things that have actually come up or would come up when running this code at scale.

1. **Label smoothing is silently disabled for cross-entropy.** You can pass `--label_smoothing 0.1 --loss_type cross_entropy` and it will accept the flag, print a warning, and zero it out. If you want label smoothing, you must use `bce` or `bce_with_logits`. Easy to miss.

2. **BCE loss scaling is not what you expect from stock PyTorch.** `nn.BCEWithLogitsLoss(reduction="mean")` averages over both batch and class dimensions. For 1000 classes, loss is ~1000× smaller than CE. This repo fixes it by using `reduction="sum"` and dividing by batch size only. If you ever swap in vanilla PyTorch BCE, your LR will be wrong by orders of magnitude.

3. **Switching loss type means retuning LR.** CE and BCE produce different gradient magnitudes even after the scaling fix. Do not assume an LR that works for `cross_entropy` transfers to `bce_with_logits`. Always rerun the LR finder or do a manual sweep.

4. **Mixup/CutMix flags without the loss branch = wasted compute.** The callback mutates the batch, but the mixed-label loss in `training_step` is commented out. CutMix is especially dangerous because it modifies images **in place** — the network sees mixed images but the loss uses the original single label. Either uncomment and verify the loss path, or do not pass `--mixup_alpha` / `--cutmix_alpha`.

5. **Train accuracy is meaningless with active mixing.** Even if you fix gotcha #4, `argmax == hard_label` accuracy does not align with a blended target. Use `val_acc` and `val_loss` (computed on clean images) as the real signal.

6. **LR finder runs on 1 GPU; training may run on N.** The finder uses a single device and the per-GPU batch size. When DDP launches with 4 GPUs, global batch is 4×. The optimal LR for a 4× larger batch is generally higher — linear scaling (multiply LR by N) with warmup is the standard heuristic, but it is not automatic.

7. **Zero-gamma BN init + downsample BN zero = aggressive.** This code zeros **both** the residual-path BN gamma and the downsample-path BN gamma. Some recipes only zero the residual side. If you see the network not learning at all in the first epoch, this double-zero might be too aggressive for your LR and warmup settings.

8. **`torch.compile` first-epoch slowness.** The first few steps (sometimes the first epoch) will be significantly slower due to tracing and compilation. Do not judge throughput or kill the job based on step 1. Memory may also spike temporarily during compilation.

9. **`torch.compile` + checkpoint resume.** Compiled models can have issues when loading checkpoints that were saved from a non-compiled (or differently compiled) run. Watch for unexpected recompilations or errors on resume.

10. **BatchNorm statistics are per-GPU in DDP.** Each rank maintains its own running mean/var. With a small per-GPU batch (say batch_size=32 across 8 GPUs), each rank's BN stats are estimated from only 32 samples — noisy. This shows up as a gap between training metrics and validation metrics, or as jittery validation curves. **SyncBatchNorm** (Lightning: `pl.utilities.model_helpers.convert_to_sync_batchnorm(model)`) fixes this but adds communication cost.

11. **`pin_memory=True` with too many workers can backfire.** Pinned memory is allocated outside the normal CUDA allocator. If `num_workers` is high and each worker pins its output, host memory pressure can cause slowdowns or OOM on the CPU side, not the GPU side. Profile if you see unexpected hangs.

12. **EarlyStopping `strict=False` hides missing metrics.** If `val_loss` is not logged for some reason (bad validation data, metric name typo), training will **not** crash — it will just never early-stop. You could train for all `max_epochs` without realizing validation was broken.

13. **Effective batch printout ignores accumulation.** The script prints `batch_size × GPU_count` as the effective batch. If you later add `accumulate_grad_batches > 1`, the real effective batch is larger than what is printed. You need to reason about this yourself.

14. **RandAugment on TinyImageNet (64×64) uses the same magnitude (9) as ImageNet (224×224).** Geometric transforms at magnitude 9 can distort a 64×64 image much more aggressively than a 224×224 one. If TinyImageNet accuracy is unexpectedly poor, try reducing `magnitude` or `num_ops`.

15. **Checkpoint `save_on_train_epoch_end=False`.** Checkpoints are saved after **validation**, not after the training epoch. If validation is slow or crashes, you lose the training progress from that epoch. Generally the right behavior, but worth knowing.

---

## File map

| Area | Where |
|------|-------|
| Trainer, device selection, LR finder, callbacks | `lightning_main.py` |
| Lightning module (loss, schedule, logging, grad norms) | `lightning_main.py` → `ImageNetLightningModule` |
| ResNet-50 v1.5, init, zero-gamma | `model_resnet50.py` |
| BCE losses, scaling fix, bias init | `bce_loss.py` |
| Mixup/CutMix callback | `augmentation.py` |
| Example run commands | `Readme_lightning.md` |
| CUDA/env tuning (used from vanilla `main.py`, not Lightning) | `gpu_optimizations.py` |
