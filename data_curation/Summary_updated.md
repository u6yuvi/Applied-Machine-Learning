1. Frame Sampling 
    1. Scene Change Detection 
        1. Optical Flow
    2. Quality Checks/Gates
        1. Blur Detection (Laplacian Variance): It calculates the "sharpness" of the image. If a frame is caught mid-motion blur or the lens is dirty, it is rejected.
        2. Luminance Check: If a frame is too dark (underexposed) or too bright (blown out/glare from factory lights), it is discarded.
Why here? Rejecting these early saves you the cost of running expensive Auto-Labeling (Stage 4) on images a human couldn't even label correctly.



In this factory scenario—characterized by long stretches of "dead time" (empty floors) and bursts of activity (forklifts, workers, deliveries)—here is the step-by-step lifecycle of how your data is curated.

---

### Step 1: The "Dead Time" Filter (Idle State)
*   **The Situation:** It’s 2:00 AM. The lights are dimmed, and nothing is moving. 
*   **How Stage 1 handles it:**
    1.  **SSIM Comparison:** The system compares Frame A to Frame B. Because nothing moved, the SSIM (Structural Similarity) is 0.999. The system says: *"This is the same image,"* and **discards it.**
    2.  **Optical Flow:** The motion vectors are near zero. The system confirms: *"No motion,"* and **discards it.**
    3.  **The Heartbeat (Max-Interval):** Every 10 minutes, the clock hits the `max_interval`. Even though nothing moved, the system **saves one frame.** 
*   **Result:** Instead of 18,000 frames of an empty dark floor (at 30fps), you have exactly **1 high-quality "night-time empty floor" image.**

### Step 2: The "Event Trigger" (Activity Starts)
*   **The Situation:** At 8:00 AM, a forklift enters the frame and a worker walks across the floor.
*   **How Stage 1 handles it:**
    1.  **SSIM Drop:** Suddenly, the similarity drops from 0.99 to 0.85 because a large yellow object (forklift) appeared.
    2.  **Optical Flow Spike:** The system detects a cluster of pixels moving from left to right.
    3.  **The Action:** The system says: *"Something is happening!"* and **saves the frame.**
*   **The Min-Interval Guardrail:** The forklift takes 10 seconds to cross. Without a guardrail, you’d save 300 frames of that one forklift. If your `min_interval` is set to 1.0s, the system saves the forklift at the entrance, then waits 1 second, saves it in the middle, waits 1 second, and saves it exiting.
*   **Result:** You capture **10 distinct images** of the event instead of 300 nearly identical ones.

### Step 3: The "Quality Gate" (Cleaning the Event)
*   **The Situation:** One of those forklift frames was captured while the camera was vibrating, making it blurry. Another was captured just as a bright overhead light reflected off the forklift's windshield (lens flare).
*   **How the Quality Gates handle it:**
    1.  **Laplacian Variance:** The system calculates the "sharpness" of the blurry frame. It falls below the threshold. **Discarded.**
    2.  **Luminance Check:** The frame with the lens flare is detected as "blown out" (too many pure white pixels). **Discarded.**
*   **Result:** You are left with only the **sharpest, most balanced images** of the forklift.

### Step 4: The "Contextual Dedup" (Stage 2)
*   **The Situation:** You have 50 cameras across the factory. Two cameras see the **same forklift** from different angles, or the same forklift stops for 2 minutes to pick up a pallet.
*   **How Stage 2 handles it:**
    1.  **pHash:** The system looks at the "fingerprint" of the images. It sees that the forklift sitting still for 2 minutes generated 12 "Heartbeat" frames that all look almost identical.
    2.  **LSH Index:** It groups these identical "sitting still" images and **deletes the duplicates**, keeping only one.
*   **Result:** Your dataset size shrinks significantly, leaving only unique "states" of the factory.

### Step 5: Diversity & Rarity (Stage 3)
*   **The Situation:** Over a month, you've captured 5,000 images of forklifts but only 3 images of a "fire extinguisher" being moved.
*   **How Stage 3 handles it:**
    1.  **K-Center Coreset:** It looks at the "Embeddings" (the mathematical meaning) of all images. It realizes that the 5,000 forklift images are very "close" to each other in mathematical space.
    2.  **Selection:** It picks a representative sample of the forklifts (e.g., 500 instead of 5,000) but **prioritizes** the "outliers" (the fire extinguisher or a rare spill on the floor) because they are mathematically "far" from the common data.
*   **Result:** A perfectly balanced dataset that doesn't over-represent common events.

### Step 6: The "Final Labeling" (Stage 4 & 5)
*   **The Situation:** You now have 1,000 high-quality, unique images.
*   **How it finishes:**
    1.  **Auto-Labeling:** Grounding DINO looks at your "Forklift" images and automatically draws a box around them.
    2.  **Human Review:** You spend 10 minutes confirming the boxes are correct.
    3.  **YOLO Export:** You get a `.zip` file ready to train your model.

---

### Comparison of Results:
| Method | Frames Collected (1 Day) | Data Quality | Training Result |
| :--- | :--- | :--- | :--- |
| **Naive (Save Everything)** | 2.5 Million | 99% Boring / Blurry | Overfit to empty floors; high false positives. |
| **Standard Time Sampling** | 86,400 (1 per sec) | 90% Boring | Still too much redundant data. |
| **This Pipeline** | **~2,000** | **100% Interesting** | Robust model that sees forklifts in all lights/angles. |



Step-3
Stage 3 is the "Intelligence" of the pipeline. Its job is to make sure your final dataset isn't just a collection of random images, but a **mathematically optimized representation** of every possible scenario in your factory.

Here is a breakdown of how **Greedy K-Center Coreset Selection** works and why those specific features matter.

---

### 1. The Core Concept: Greedy K-Center
Imagine all your images are dots on a map. "Similar" images (like two different forklifts) are close together. "Different" images (a forklift vs. a person) are far apart.

If you need to pick 1,000 images out of 100,000, you don't want 1,000 dots clustered in one corner. You want to spread your "picks" across the whole map so that **no point on the map is too far from a picked image.**

**How the algorithm works step-by-step:**
1.  **Start:** Pick one image (either randomly or a specific "Seed").
2.  **Calculate Distances:** For every other image in your pool, calculate how far it is from the one you just picked.
3.  **The "Greedy" Choice:** Find the image that is **farthest away** from your first pick. Add it to your selection.
4.  **Repeat:** Now find the image that is farthest from its *nearest* already-selected neighbor. 
5.  **Stop:** Keep going until you have $K$ images (your target count).

**The Result:** You end up with a "Coreset"—a small subset that covers the "diversity" of the entire large set. The "2-approximation" mention is a mathematical proof that this simple greedy method is nearly as good as the most complex possible optimization.

---

### 2. Stratified Quotas (The "Safety" Rule)
In a factory, 95% of your data might be "Normal Operations." Only 5% might be "Rainy Day" or "Night Shift." 

If you run a pure K-Center selection, the algorithm might still overlook the "Night Shift" because there are so few examples. 
*   **How it works:** You tag images with attributes (e.g., `lighting: night`). 
*   **The Action:** You tell the algorithm: "I want 10,000 images total, but **at least 2,000 must be from the Night Shift.**"
*   **Why it's useful:** It prevents the model from being "blind" to rare but important conditions.

---

### 3. Per-Class Minimums (The "Object" Rule)
This usually happens **after** Stage 4 (Auto-Labeling). After the AI has "guessed" what is in the images (forklifts, people, etc.), you might find you have:
*   50,000 images of "Boxes."
*   200 images of "Fire Extinguishers."

If you just pick the most "diverse" images, you might accidentally throw away half of your rare fire extinguisher images because they look "visually similar" to red boxes.
*   **The Action:** You set a floor: "Keep at least 5,000 forklift images and **100% of fire extinguisher images**."
*   **Why it's useful:** It ensures the YOLO model has enough examples of every specific object class to actually learn them.

---

### 4. Seed Indices (The "Knowledge" Rule)
Sometimes you already have a "Golden Dataset"—1,000 perfect images you labeled last year. You don't want to pick new images that look exactly like the ones you already have.

*   **How it works:** You provide the "Seed Indices" (the IDs of your existing images) to the algorithm.
*   **The Action:** The algorithm treats those seeds as "already picked." It starts searching for the **farthest** images relative to your existing library.
*   **Why it's useful:** This is how you **grow** a dataset over time. Instead of just adding more data, you only add data that "teaches" the model something it doesn't already know.

---

### Summary: Why this is better than Random Sampling
If you just pick 10,000 random images from a factory:
1.  **Redundancy:** You’ll have 5,000 images of the same "Person A" at the same "Workstation B."
2.  **Blind Spots:** You’ll likely have 0 images of a "Spill on the Floor" because spills are rare.

If you use **Greedy K-Center Diversity Selection**:
1.  **Efficiency:** Every image in the 10,000 is visually unique.
2.  **Robustness:** Because it seeks out "farthest points" (outliers), it is much more likely to catch rare events, weird camera glitches, or unusual floor activity that random sampling would miss. 

**Think of it as: Instead of buying 100 random books, you are carefully picking one book from every single genre in the library.**

Yes, there are algorithms and specific Python implementations designed to handle millions of images. 

To process **1 million frames**, a naive Python `for` loop is too slow because it would require $10^{11}$ (100 billion) distance comparisons if you select 100,000 images ($N=10^6, K=10^5$). 

Here are the best ways to run this at scale:

### 1. The Optimized Greedy K-Center (Gonzalez's Algorithm)
This is the industry standard for "farthest-first" diversity selection. To make it run on a million images, you must avoid recalculating the entire distance matrix. Instead, you maintain a **"Running Min-Distance"** vector.

*   **Complexity:** $O(N \times K)$
*   **The Scalability Trick:** You only calculate the distance from all $N$ points to the **one** newly added center in each step.
*   **Implementation:** Use **PyTorch on a GPU** to vectorize the distance calculations. A GPU can process millions of distance updates per second.

#### High-Performance Code Pattern (PyTorch/GPU)
```python
import torch

def k_center_greedy_gpu(embeddings, count, seeds=None):
    # embeddings: (N, D) tensor on GPU
    # count: Number of frames to select
    N, D = embeddings.shape
    min_distances = torch.full((N,), float('inf')).to(embeddings.device)
    selected_indices = []

    # 1. Initialize with Seeds or a random point
    if seeds is not None:
        selected_indices.extend(seeds)
        # Update distances for all seeds at once
        seed_embeds = embeddings[seeds] # (S, D)
        dists = torch.cdist(embeddings, seed_embeds) # (N, S)
        min_distances = torch.min(dists, dim=1).values
    else:
        first_idx = torch.randint(0, N, (1,)).item()
        selected_indices.append(first_idx)
        min_distances = torch.norm(embeddings - embeddings[first_idx], dim=1)

    # 2. Greedy Loop
    for _ in range(len(selected_indices), count):
        # Pick the point farthest from its nearest selected neighbor
        new_idx = torch.argmax(min_distances).item()
        selected_indices.append(new_idx)
        
        # Update distances: only check distance to the NEWLY added point
        new_dists = torch.norm(embeddings - embeddings[new_idx], dim=1)
        min_distances = torch.min(min_distances, new_dists)
        
    return selected_indices
```

### 2. Large-Scale Libraries
If you don't want to write it from scratch, use these specialized libraries:

*   **[DeepCore](https://github.com/PatrickZH/DeepCore):** An open-source library specifically for coreset selection in deep learning. It contains a highly optimized `kCenterGreedy` class used for pruning massive datasets like ImageNet.
*   **[FAISS](https://github.com/facebookresearch/faiss):** While known for K-Means, you can use FAISS's `IndexFlatL2` to perform the "farthest point" search. Since FAISS is written in C++ with GPU support, it is the fastest way to handle 1M+ vectors.
*   **[Submodlib](https://github.com/decile-team/submodlib):** This library implements "Facility Location" (which is the mathematical generalization of k-center). It is designed for data subset selection and diversity.

### 3. Handling Your Specific Constraints
To handle a million images with your industrial constraints, you "anchor" the greedy search:

1.  **Seed Indices:** As shown in the code above, you pre-populate the `selected_indices` with your rare or known images. The algorithm will then naturally move as far away from those as possible, ensuring the new data is truly "novel" compared to what you already have.
2.  **Stratified Quotas:** You run the algorithm in **batches**. If you need 2,000 "Night Shift" images, you filter your 1M pool to just "Night Shift" images and run the K-center on that sub-pool first.
3.  **Per-Class Minimums:** 
    *   First, select all images of very rare classes (e.g., a "Fire" or "Accident" class).
    *   Treat these as your **Seed Indices**.
    *   Run the K-center algorithm to fill the rest of the $K$ slots. This ensures you keep the rare classes while getting maximum diversity in the common classes (like "Empty Floor").

### Summary of Performance for 1M Images
| Method | Speed | Recommendation |
| :--- | :--- | :--- |
| **Standard Python** | Days | **Avoid.** |
| **PyTorch (GPU)** | 10–30 Minutes | **Best for custom pipelines.** |
| **FAISS (GPU)** | 2–5 Minutes | **Best for pure speed.** |
| **DeepCore/Submodlib** | Varies | **Best if you want research-grade diversity.** |