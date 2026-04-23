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