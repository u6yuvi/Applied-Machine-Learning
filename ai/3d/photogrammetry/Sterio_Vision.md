This guide covers **Stereo Vision** (why two views are better than one) and **Orientation** (how we place reconstructed 3D models into real-world coordinates).

---

# Part 1: Stereo Vision - The Power of Two

If you close one eye, it is hard to judge exactly how far away a ball is. Open both eyes, and your brain triangulates distance. Photogrammetry does the same using two cameras (or two viewpoints).

### 1. The "Normal Case" (Perfect Stereo)

In the normal case, the two cameras are ideally side by side, at the same height, and pointing in the same direction.

- **Intuition:** The major difference between left and right images is a shift in the **X direction**.
- **Key term - Disparity:** The difference in a feature's pixel position between the left and right images.
  - **Close objects** have **high disparity** (large shift).
  - **Far objects** have **low disparity** (small shift).
- **Benefit:** Matching is easier because correspondences are searched along horizontal lines.

### 2. Triangulation (Finding the 3D Point)

After finding the same feature in both images, draw one ray from each camera center through the matched image point.

- **Problem:** In noise-free geometry, the rays intersect at the 3D point. In practice, due to noise/model error, rays often do not intersect.
- **Solution:** Estimate the 3D point closest to both rays (often the midpoint of the shortest segment between them).
- **Uncertainty field:**
  - **Near the camera:** Rays intersect at a larger angle, so depth is more stable.
  - **Far from the camera:** Rays become nearly parallel, so tiny pixel errors can cause large depth ($Z$) errors.

---

# Part 2: Orientation - Putting It on the Map

After reconstructing from images, you may get a good 3D shape, but the model can still be floating in an arbitrary frame: unknown scale, location, and global orientation.

### 1. Relative Orientation (The "Handshake")

- **What it is:** Camera A and Camera B are solved **relative to each other**.
- **Result:** A valid 3D model in a local coordinate system, but without true GPS position or real-world scale.

### 2. Absolute Orientation (The "Reality Check")

- **What it is:** Mapping that local model to the real object/world frame.
- **Math:** A **7-DoF similarity transform**:
  - **3 translations** (place model at correct location).
  - **3 rotations** (align model heading/attitude).
  - **1 scale** (set model units to real units, e.g., meters).
- **Control points:** Use **Ground Control Points (GCPs)** with known coordinates to lock the transform (minimum 3 non-collinear points in practice for a stable fit).

---

# Part 3: The Toolkit (How to Solve It)

### 1. DLT (Direct Linear Transformation)

- **Best for:** Cases where camera intrinsics are unknown.
- **Requirement:** At least **6 control points**.
- **Catch:** Points must span 3D; purely coplanar points make the solution unstable.

### 2. P3P (Perspective-3-Point)

- **Best for:** When intrinsics ($K$) are already known and you need camera pose.
- **Requirement:** 3 points produce up to 4 candidate poses, then 1 extra point disambiguates.

### 3. Bundle Adjustment (The Gold Standard)

- **What it is:** Joint nonlinear optimization of all camera poses and 3D points.
- **Goal:** Minimize **reprojection error** (difference between observed pixels and projected 3D points).
- **Practical note:** Needs a good initialization; poor initial guesses can diverge or converge to bad local minima.

---

# The Playbook: Which Strategy to Choose?

| Scenario | Strategy | Why? |
| :--- | :--- | :--- |
| **I have 2 photos and no GPS info.** | **Relative Orientation + Triangulation** | Build geometry first, then georeference later. |
| **I have a drone with known camera ($K$) flying over a known area.** | **P3P per frame** | Efficient camera tracking with known intrinsics. |
| **I have old CCTV footage and unknown lens zoom/intrinsics.** | **DLT** | Solves camera model/pose from control correspondences. |
| **I want maximum 3D accuracy.** | **Bundle Adjustment** | Globally optimizes all variables together. |
| **I want better error diagnosis.** | **Two-step solution** | Solve camera geometry first, then GPS mapping to isolate error sources. |

---

# Real-World Case Studies

### 1. Security Camera (Single View)

- **Problem:** A crime appears in one camera; estimate suspect height.
- **Solution:** Without stereo, use **absolute orientation** and known-size references (control points) to recover scale, then measure height.

### 2. Smart Doorbell (Stereo)

- **Problem:** Distinguish a real person from a flat printed photo spoof.
- **Solution:** Use stereo disparity/depth: a real 3D person has meaningful depth variation; a flat photo has near-zero internal depth.

### 3. Industrial Robotics (Robot Arm)

- **Problem:** Robot must pick a part from a bin reliably.
- **Solution:** Use **eye-in-hand calibration** and **P3P** for pose estimation, then refine scene geometry with **bundle adjustment** so grasp plans avoid collisions.

---

# "Gotchas" (Deep Concepts)

**Q: Why is Bundle Adjustment considered statistically optimal (under common assumptions)?**  
- **Answer:** It optimizes all observations jointly, typically via least squares (or robust variants), distributing noise across the full system instead of overfitting individual measurements.

**Q: What is the 7th degree of freedom in absolute orientation?**  
- **Answer:** **Scale.** Pure image geometry cannot tell toy-car scale from real-car scale without a metric reference.

**Q: Why do rays often fail to intersect in triangulation?**  
- **Answer:** Pixel noise, imperfect distortion correction, and small pose errors all cause skew rays in practice.

**Q: Can DLT be solved from a flat checkerboard only?**  
- **Answer:** Not robustly for full 3D projection estimation; coplanar control points make DLT poorly conditioned for general 3D recovery.

This is an **ultimate photogrammetry and computer vision playbook** that bridges classic geometric methods and modern AI-driven pipelines.

---

# Part 4: Advanced Theory

### 1. Epipolar Geometry (Line-Constrained Search)

In ideal stereo we assume parallel cameras, but real camera pairs are often tilted and offset. Epipolar geometry handles the general case.

- **Fundamental matrix ($F$):** Relates pixels between two uncalibrated images; does not require intrinsics.
- **Essential matrix ($E$):** Calibrated relation between normalized rays, with $E = K'^T F K$.
- **Intuition:** A point in image A maps to an **epipolar line** in image B; its match must lie on that line.
- **Why it matters:** Reduces search from 2D area to 1D line (epipolar constraint), improving robustness and speed.

### 2. Collinearity Equation (The Core Physical Law)

Nearly all photogrammetry software is built on this principle:

- **Concept:** Camera center, image point, and object point are collinear.
- **Interview framing:** Bundle adjustment iteratively updates camera and point parameters to satisfy collinearity across all observations.

### 3. Feature Matching (Finding Correspondences)

How does the system find matches?

- **Detection:** Find salient structures (corners, blobs, etc.).
- **Description:** Encode local appearance as a descriptor vector ("fingerprint").
- **Traditional methods:** **SIFT** (accurate, slower), **ORB** (faster, common in robotics).
- **Modern methods:** **SuperPoint** and **LoFTR**, often more robust in low texture, blur, or difficult lighting.

---

# Part 5: SfM vs. SLAM (Systems Level)

| Feature | SfM (Structure from Motion) | SLAM (Simultaneous Localization and Mapping) |
| :--- | :--- | :--- |
| **Goal** | Highest quality 3D reconstruction. | Real-time localization + mapping for navigation. |
| **Data** | Batch photo set. | Streaming video/sensor input. |
| **Speed** | Slower (minutes/hours/days). | Fast (milliseconds to real-time). |
| **Logic** | Global/batch optimization over many views. | Incremental updates on current + recent states. |
| **Example** | 3D model of a cathedral for mapping/archiving. | Autonomous driving on a live roadway. |

---

# Cheat Sheet of Failures (When Things Go Wrong)

| Technique | When it fails | Why? |
| :--- | :--- | :--- |
| **Checkerboard calibration** | Out-of-focus or motion blur frames. | Corner detection becomes unreliable. |
| **Stereo vision** | Flat white wall, glass, or repetitive low texture. | Few reliable matches between views. |
| **DLT** | All control points lie on one plane. | Degenerate/ill-conditioned geometry. |
| **P3P** | Three points nearly collinear. | Pose ambiguity and numerical instability. |
| **Bundle Adjustment** | Bad initialization. | Convergence to poor local minima or divergence. |

---

# Part 6: Modern Industry Strategy Playbook

The field has moved beyond pure classical geometry. This is the practical trend in industry (Meta/Tesla/Google-like stacks):

| Task | Classic solution | Modern preferred approach | Why the change? |
| :--- | :--- | :--- | :--- |
| **Calibration** | Checkerboard (Zhang) | **Self-calibration** | Devices can refine intrinsics/extrinsics from natural motion and scene observations. |
| **Feature matching** | SIFT/SURF | **Deep features (e.g., LoFTR)** | Better robustness under lighting, viewpoint, and appearance changes. |
| **3D reconstruction** | Triangulation + meshing | **Gaussian Splatting / NeRF** | Strong photorealism and view synthesis quality. |
| **Depth sensing** | Stereo cameras | **Monocular depth (AI)** | Learned priors infer depth from a single camera in many scenarios. |
| **Optimization back-end** | Bundle adjustment only | **Factor graphs** | Incremental updates are easier for online robotics and sensor fusion. |

---

# "Gotchas" (Interview + Practical)

**Q: What is sub-pixel accuracy, and why do we need it?**  
- *Answer:* Pixel-level corner detection (e.g., $(10,10)$) is coarse. Sub-pixel localization (e.g., $(10.42,10.18)$) greatly improves geometric precision and can boost 3D accuracy significantly.

**Q: When is a homography ($H$) better than a fundamental matrix ($F$)?**  
- *Answer:* Use **homography** for planar scenes or pure rotation. Use **fundamental matrix** for general 3D scenes with depth variation.

**Q: A high-mounted security camera must measure car speed. How?**  
- *Answer:* (1) Use **DLT** or **absolute orientation** to map image pixels to ground-plane meters. (2) Measure frame-to-frame pixel displacement. (3) Convert to meters and divide by elapsed time (from frame rate/timestamps).

**Q: Why correct radial distortion before triangulation?**  
- *Answer:* Triangulation assumes pinhole straight-ray geometry. Distorted image points violate that assumption and produce biased or inconsistent 3D intersections.

---