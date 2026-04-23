# Data Curation Pipeline
End-to-end pipeline for curating high-quality object detection datasets from continuous camera feeds. Designed for factory-floor / industrial use cases where you have millions of raw frames and need a clean, diverse, labeled dataset for YOLO training.
## Pipeline Overview
```Raw video feeds (10M+ frames) │ ├─ Stage 1: Frame Sampling ──────────── 10M → 500K (scene-change detection) ├─ Stage 2: Deduplication ───────────── 500K → 80-150K (pHash + DINOv2 embeddings) ├─ Stage 3: Diversity Selection ─────── 150K → 100K (k-center coreset) ├─ Stage 4: Auto-Labeling ───────────── 100K annotated (Grounding DINO + SAM2) ├─ Stage 5: Human Review ────────────── prioritized corrections (Label Studio) └─ Stage 6: Dataset Assembly ────────── YOLO format, train/val/test splits```
## Project Structure
```data_curation/├── Readme.md└── src/ ├── stage1.py # SceneChangeFrameSampler — SSIM + optical flow + quality gates ├── stage2.py # PerceptualHashIndex (LSH) + embedding-based dedup ├── stage3.py # Greedy k-center coreset selection + stratified sampling ├── stage4.py # Auto-labeling with Grounding DINO, SAM2, CLIP re-scoring ├── stage5..py # Human review queue with active-learning prioritization └── stage6.py # Orchestrator — runs full pipeline end-to-end```
## Stage Details
### Stage 1: Intelligent Frame Sampling (`stage1.py`)
Reduces raw frames by 90-95% using scene-change detection instead of naive time-interval sampling.
- **SSIM** between consecutive frames to detect scene changes- **Optical flow** magnitude to catch motion even when SSIM stays high- **Quality gates**: rejects too-dark, blown-out, or blurry frames (Laplacian variance)- **Time guardrails**: min interval (anti-burst) and max interval (catch slow changes)- **Perceptual hash** computed per frame for downstream cross-camera dedup
Key classes: `SceneChangeFrameSampler`, `SamplingConfig`, `FrameMetadata`
### Stage 2: Perceptual Deduplication (`stage2.py`)
Two-tier dedup that avoids O(N²) pairwise comparison:
1. **Fast pass** — perceptual hash (pHash) with LSH index for approximate nearest neighbor. O(N log N), handles 500K images in ~5 seconds.2. **Precise pass** — DINOv2/CLIP embedding cosine similarity on hash-collision clusters only.
Diversity guardrails ensure minimum frames per hour-slot and per camera.
Key classes: `PerceptualHashIndex`, `DedupConfig`
### Stage 3: Diversity Selection (`stage3.py`)
From the deduplicated pool, selects the most diverse K frames via **greedy k-center coreset selection** — iteratively picks the point farthest from all already-selected points (2-approximation to optimal k-center).
Supports:- **Stratified quotas** by scene attributes (night, rain, etc.)- **Per-class minimums** after auto-labeling (e.g., ensure ≥5000 forklift images)- **Seed indices** to anchor selection on known rare examples
Key function: `coreset_selection_greedy_kcenter`
### Stage 4: Auto-Labeling (`stage4.py`)
Foundation-model-based annotation generation:
1. **Grounding DINO** — open-vocabulary detection with text prompts per class2. **SAM2** — instance segmentation masks on detected boxes3. **CLIP** — confidence re-scoring for ambiguous detections4. Exports to **YOLO format** (one `.txt` per image)
Confidence routing:- `> 0.85` → auto-accept (no human review needed)- `0.40 – 0.85` → sent to human review queue- `< 0.40` → discarded
Default classes: `person`, `forklift`, `car`, `cart`, `stop_sign`, `dock_door`
Key function: `auto_label_batch`
### Stage 5: Human Review (`stage5..py`)
Prioritized review queue using active learning signals:
- **Model uncertainty** — low-confidence predictions reviewed first- **Novel scenes** — high embedding distance from existing training data- **Class imbalance** — boost priority for underrepresented classes
Estimated human effort for 100K images:- ~60K auto-accepted (no review)- ~30K quick verification (~2 sec/image)- ~10K full correction (~30 sec/image)- **Total: ~100 hours** (~2 annotators × 1 week)
Designed for export to **Label Studio** (self-hosted, data stays on-prem).
Key function: `create_review_queue`
### Stage 6: Orchestrator (`stage6.py`)
Runs the full pipeline end-to-end with CLI arguments:
```bashpython src/stage6.py \ --video-dir /mnt/security_cameras/ \ --output-dir /home/jovyan/datasets/factory_100k/ \ --target-count 100000 \ --classes person forklift car cart stop_sign dock_door```
Flags: `--skip-sampling`, `--skip-dedup`, `--skip-auto-label` to resume from intermediate stages.
Outputs a timestamped run directory with `data.yaml` and `images/{train,val,test}` + `labels/{train,val,test}` (80/10/10 split).
## Dependencies
- `opencv-python` — frame I/O, optical flow, image processing- `scikit-image` — SSIM computation- `numpy` — array operations, coreset algorithm- `faiss-cpu` or `faiss-gpu` — production LSH / embedding index (optional, replaces built-in LSH)- `torch` + `transformers` — DINOv2, Grounding DINO, CLIP, SAM2
## Industry Benchmarks
| Metric | Value | Source ||--------|-------|--------|| Scene-change sampling reduction | 90-95% frame reduction | Tesla AI Day 2022, Waymo Open Dataset || Perceptual dedup reduction | 40-60% further reduction | Scale AI benchmarks || Auto-label accuracy (Grounding DINO) | 75-85% mAP on custom domains | IDEA Research || Human review time saved | 70-90% vs from-scratch labeling | Scale AI, V7 Labs || Human review speed (verify) | 2-3 sec/image | Labelbox benchmarks || Human review speed (correct) | 20-40 sec/image | CVAT user studies || k-center coreset diversity gain | 5-15% mAP improvement vs random at same size | Google Research 2020 |
## Recommended Tool Stack
| Component | Tool ||-----------|------|| Frame extraction | FFmpeg + OpenCV (this pipeline) || Deduplication | FAISS + DINOv2 embeddings || Auto-labeling | Grounding DINO + SAM2 (open source) || Human review | Label Studio (self-hosted, on-prem) || Dataset management | FiftyOne (Voxel51, open source) || Version control | DVC (Data Version Control) || Training | Ray + Ultralytics (see `../ray/`) |