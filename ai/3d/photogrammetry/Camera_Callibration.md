# Camera calibration: mapping world points to pixels

In photogrammetry, we use a mathematical model to describe how a 3D point in the real world ($\mathbf{X}$) ends up as a 2D coordinate on a digital image ($\mathbf{x}$).

### The master equation

$$\mathbf{x} = P\mathbf{X} \implies \text{Pixel} = K [R \mid t] \mathbf{X}_{\text{world}}$$

*Where **$P$** is the **projection matrix**, **$K$** is the **intrinsic** (calibration) matrix, and **$[R \mid t]$** encodes the **extrinsic** pose (rotation $R$ and translation $t$). In homogeneous coordinates, $\mathbf{X}$ is a 4-vector (often with a trailing 1) so that projection stays a single matrix multiply.*

---

## 1. The four coordinate systems (the journey of a point)

To map a point, we move it through four distinct “universes”:

1. **World / object coordinates ($X_W, Y_W, Z_W$):** A fixed 3D reference (e.g., GPS coordinates or the corner of a room).
2. **Camera coordinates ($X_C, Y_C, Z_C$):** A 3D system where the **optical center** of the camera is $(0,0,0)$. The $Z$-axis usually points **out** of the lens (along the optical axis).
3. **Image plane coordinates ($x, y$):** A 2D system in **physical** units (e.g., millimeters)—where the pinhole model says light intersects the “film” plane.
4. **Sensor / pixel coordinates ($u, v$):** The final digital grid in **pixels**. The origin $(0,0)$ is typically the **top-left** of the image.

---

## 2. Extrinsic parameters (the pose)

**Definition:** Extrinsics describe the camera’s **position** and **orientation** in the world.

* **Total parameters:** **6** (3 for translation, 3 for rotation).
* **Transformation (concept):** A common form is $X_{\text{camera}} = R(X_{\text{world}} - C)$, where $C$ is the **camera center** expressed in world coordinates. This is equivalent to the block $[R \mid t]$ with $t = -RC$ when you write $X_{\text{camera}} = R X_{\text{world}} + t$.
* **Intuition:** Calibrating or estimating extrinsics means answering: *“Where is the camera, and what is it looking at?”*

To understand the **three heading / rotation parameters** (often called **Euler angles**), it helps to imagine the camera as an airplane—or even your own head. These three parameters define which direction the camera is “facing” in 3D space. In photogrammetry and computer vision, we usually label them **pitch**, **yaw**, and **roll**.

#### A. Pitch (tilt)

* **The action:** Tilting the camera **up or down**.
* **The axis:** Rotation around the **$X$-axis** (a horizontal line left–right through the camera).
* **Example:** A security camera looking down at a parking lot has a downward pitch. In robotics, if a drone leans forward to move, it is pitching.

#### B. Yaw (pan)

* **The action:** Turning the camera **left or right**.
* **The axis:** Rotation around the **$Y$-axis** (a vertical line through the camera).
* **Example:** Someone at a window scanning from the far left to the far right of the street. In a car, when you turn a corner, a dashboard camera yaws.

#### C. Roll (bank)

* **The action:** Tilting the camera **side to side** (as if touching an ear to a shoulder).
* **The axis:** Rotation around the **$Z$-axis** (the optical axis, straight out of the lens).
* **Example:** Holding a phone slightly crooked introduces roll. In photogrammetry, roll is often annoying to fix because it makes the horizon look slanted.

---

### The mathematical form: the rotation matrix ($R$)

In the pipeline $\mathbf{x} \sim P \mathbf{X}_{\text{world}}$, the three angles are encoded in a **$3 \times 3$ rotation matrix $R$**.

* **Orthonormal:** Rows and columns are unit length and mutually perpendicular. That way, rotation **preserves lengths and angles**—the rigid camera does not stretch or squash the scene.
* **Order matters:** Composing rotations (e.g., yaw then pitch vs. pitch then yaw) generally **does not commute**. Pipelines must fix an **Euler angle order** (e.g., $XYZ$ vs. $ZXY$). A related pitfall is **gimbal lock** (see interview notes below): near certain configurations, two rotation axes align and one degree of freedom becomes ill-conditioned when using Euler angles.

---

### Real-world examples of heading / rotation

| Use case | Dominant rotation parameter | Why it matters |
| :--- | :--- | :--- |
| **Fixed security camera** | **Pitch & yaw** | The camera is bolted to a wall. You pan (yaw) toward the door and tilt (pitch) toward the floor. Roll is often kept near 0. |
| **Handheld photogrammetry** | **All three (P, Y, R)** | Hand shake and tilt change every shot; structure-from-motion must estimate all three per image to stitch reliably. |
| **Drone mapping** | **Roll & pitch** | Wind and flight dynamics induce roll and pitch. Without a gimbal, software must “un-roll” imagery so the map stays consistent. |
| **Self-driving cars** | **Pitch** | Speed bumps and suspension motion pitch the camera; ignoring this can make objects appear to jump vertically in 3D. |

---

## 3. Intrinsic parameters (the “guts”)

**Definition:** Intrinsics describe **internal** camera properties that map rays through the optical center onto the **sensor grid** (focal effects, principal point, pixel sampling, and optionally skew).

### The affine camera model (five intrinsic parameters)

When mapping the image plane to the sensor, we often use an **affine** step (linear part of $K$, before distortion):

1. **Camera constant ($c$):** Often loosely called **focal length** in calibration; it is the effective distance from the optical center to the image plane in consistent units (related to field of view).
2. **Principal point ($x_h, y_h$):** Where the optical axis meets the sensor—the **optical center** in pixel coordinates. Manufacturing tolerances mean it is rarely exactly the image center.
3. **Scaling factor ($m$):** Relates physical size on the sensor to **pixels** (pixel pitch).
4. **Shear ($s$):** Models non-orthogonality between pixel axes. For modern lithographed sensors, $s$ is **usually taken as 0**.

---

## 4. The three-step mapping process (core logic)

### Step 1: Ideal perspective projection (3D → 2D)

We project $(X_C, Y_C, Z_C)$ onto the 2D image plane using similar triangles.

* **Crucial fact:** This step is **not invertible** from a single image: **depth is lost**. One pixel corresponds to a **half-line (ray)** through the camera center, not a unique 3D point.
* **Math (ideal pinhole):** $x = c \cdot (X/Z)$ (and analogously for $y$), up to the chosen axis convention.

### Step 2: Mapping to the sensor (the “affine” step)

We convert physical 2D measurements on the plane into **pixel** coordinates.

* **Shift:** Move the origin from the optical / principal point convention to the image corner (typical $u,v$ origin).
* **Scale:** Apply pixel density ($m$) and aspect ratio if needed.
* **The matrix:** These linear steps pack into the **calibration / intrinsic matrix $K$**.

### Step 3: Compensation (the “reality” step)

This is a **non-linear** refinement beyond the ideal pinhole + affine model:

1. **Lens distortion:** Real lenses bend rays; marginal rays deviate more than paraxial ones.
   * **Radial distortion:** Lines bow outward (**barrel**) or inward (**pincushion**).
2. **Sensor planarity:** A CMOS/CCD die may deviate slightly from a perfect plane.
3. **Manufacturing tolerances:** Decentering and asymmetry break the ideal model.

**Result:** A **small, position-dependent** correction moves each ideal pixel to a measured one. Calibration estimates these distortion parameters jointly with $K$ and (when applicable) extrinsics.

---

## 5. Calibration models: a complexity scale

| Model | Parameters | Components |
| :--- | :--- | :--- |
| **Unit** | 6 | Extrinsics only (position + rotation). |
| **Ideal** | 7 | Extrinsics + one focal-length-like scale. |
| **Euclidean** | 9 | Extrinsics + focal length and principal point ($x_h, y_h$). |
| **Affine** | 11 | Extrinsics + five intrinsics (focal, $x_h, y_h$, scale, skew). |
| **General** | 11 + $N$ | Affine (or Euclidean) intrinsics + $N$ **distortion** coefficients (radial / tangential terms as chosen). |

---

## 6. Real-world applications

* **Security cameras:** Often use **wide-angle** optics with noticeable **barrel** distortion. Height and metrology at the image border require **undistortion**; otherwise subjects look stretched or tilted.
* **Industrial inspection:** **Telecentric** lenses approximate **parallel** projection in object space; objects appear nearly the same size with depth. The projection matrix $P$ then differs materially from a standard perspective model.
* **Robotics (SLAM / VO):** Wrong extrinsics (even ~$1^\circ$) compound into large trajectory and map errors; intrinsics errors bias scale and reprojection.
* **Fisheye / automotive surround:** Very wide fields of view break the simple pinhole Step 1; practitioners use **fisheye** models (e.g., equidistant or equisolid) and dedicated calibration targets or motion-based bundles.

---

## Gotchas & deep-dive

### 1. The ray concept

* **Question:** “If I give you a 2D pixel coordinates, where is the 3D point?”
* **Answer:** You do not know a **unique** point—only the **ray** from the camera center through that pixel. You need a **second view** (stereo / multi-view), **depth** from a sensor, or **priors** to fix depth along the ray.

### 2. Homogeneous coordinates

* **Question:** “Why is the projection matrix $3 \times 4$?”
* **Answer:** **Homogeneous coordinates** let us write **rotation, translation, and perspective divide** as **linear operations** in lifted space. Appending a 1 to 3D points makes the world-to-camera rigid transform and the projection to the image plane composable as matrix multiplies; the final step is **dehomogenizing** (divide by the third component for $\mathbf{x}$).

### 3. Radial vs. tangential distortion

* **Radial:** Dominated by **lens shape** (strong near the image border—“fisheye-like” bending in the extreme).
* **Tangential:** Dominated by **decentering**—lens elements not perfectly aligned with the sensor normal; introduces asymmetric warping not purely radial about the principal point.

### 4. Direct linear transformation (DLT)

* **Concept:** Given **known 3D–2D correspondences**, DLT estimates the **projection matrix $P$** (up to scale) by solving a linear least-squares problem. It is a standard “first calibration” in textbooks; real pipelines refine $K$, $[R \mid t]$, and distortion with non-linear optimization (e.g., bundle adjustment).

### 5. Why is skew usually zero?

* **Answer:** Modern sensors are fabricated so pixel rows and columns are **highly orthogonal**. Skew mainly shows up in **legacy** setups (e.g., some scanned film paths) or **pathological** misalignment—not in typical CMOS phone or machine-vision cameras.

### �� Rotation-specific follow-ups

1. **“What is gimbal lock?”**  
   *Answer:* With **Euler angles**, there are configurations where two axes align and you **lose a degree of freedom** in the parameterization (singularities). That is one reason **quaternions** or **axis–angle** representations are preferred for filtering and interpolation in robotics and graphics—even though calibration outputs often still report Euler angles for humans.

2. **“Why the inverse / transpose of $R$?”**  
   *Answer:* $R$ is **orthogonal**, so $R^{-1} = R^{\mathsf{T}}$. Moving points **from world to camera** uses the rotation that expresses the world basis in camera coordinates; the **inverse** rotation maps camera vectors back to world. Sign conventions in $[R \mid t]$ vs. $R(X - C)$ must stay **consistent** in code.

3. **“Can you rotate the camera about a point that isn’t its center?”**  
   *Answer:* Not with a **pure** rotation about the lens center alone. The rigid transform is **translate → rotate → translate back** (conjugation), or equivalently a rotation about an arbitrary axis combined with translation. That is why we **define** the camera frame with the optical center at the origin for the pinhole model.

4. **“How many numbers in $R$ vs. three Euler angles?”**  
   *Answer:* $R$ has **nine** entries but only **three degrees of freedom**; the orthonormality constraints (**length and perpendicularity**) remove the extra six. Parameter counting must respect those constraints when interpreting covariance or priors.



**How do we actually find the Intrinsic matrix ($K$)?** (Calibration) and **How do we find the Camera's position if we already know $K$?** (Localization).


# Camera Calibration (Zhang’s Method)
**The Problem:** You have a camera, but you don't know its focal length, principal point, or distortion.
**The Solution:** Show the camera a known object (a checkerboard) from different angles.

### 1. The Setup: The "Z=0" Trick
Zhang’s Method is the industry standard (used in OpenCV/MATLAB).
*   **The Pattern:** A flat checkerboard. Why? Because the corners are easy for computer algorithms to find with sub-pixel accuracy.
*   **The World Origin:** We define the **World Coordinate System** $(0,0,0)$ as the top-left corner of the checkerboard.
*   **The Simplification:** Because the board is flat, every point on that board has a **$Z$ coordinate of 0**. 

### 2. The Math: From Projection to Homography ($H$)
Normally, the mapping is:
$$\begin{pmatrix} x \\ y \\ 1 \end{pmatrix} = K \begin{bmatrix} r_1 & r_2 & r_3 & t \end{bmatrix} \begin{pmatrix} X \\ Y \\ Z \\ 1 \end{pmatrix}$$
But since **$Z = 0$**, the third column of the rotation matrix ($r_3$) is multiplied by zero and disappears!
*   **The Homography ($H$):** We are left with a $3 \times 3$ matrix called a **Homography**.
    $$H = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix}$$
*   **What $H$ does:** it maps the points on the flat checkerboard plane directly to the points on the flat sensor plane.

### 3. Solving for $K$ (The Requirements)
To solve for the internal parameters ($K$), we use the properties of rotation matrices (that $r_1$ and $r_2$ are perpendicular and have a length of 1).
*   **Points per image:** You need at least **4 points** per plane to calculate $H$.
*   **The $B$ Matrix:** We define a new matrix $B$ (which contains the info for $K$). $B$ has **6 degrees of freedom** (6 unknowns).
*   **Equations per image:** Each checkerboard image provides **2 equations** to solve for $B$.
*   **Minimum Images:** To solve for 6 unknowns using 2 equations per image, you need **at least 3 different views** of the checkerboard.
*   **Final Step:** Solve the system of equations ($Vb = 0$) to extract the calibration matrix $K$.

---

# Camera Localization (P3P)
**The Problem:** You already know the camera's "guts" ($K$), and you see 3 known points in the world. **Where is the camera?**
**The Solution:** The **Perspective-3-Point (P3P)** algorithm.

### 1. How it works
P3P uses the geometry of triangles. Imagine the camera is the tip of a pyramid, and the 3 points on the ground are the base.
*   **Inputs:** The 3D coordinates of 3 points ($A, B, C$) and their 2D pixel positions.
*   **The Equation:** It creates a **fourth-degree polynomial equation**. 
*   **The Result:** Because it is a 4th-degree equation, you can get up to **4 possible mathematical solutions** (4 places the camera *could* be).

### 2. Disambiguation (The 4th Point)
Since we can't be in four places at once, we need a tie-breaker.
*   **The 4th Point:** By checking a 4th point, we see which of the 4 solutions correctly predicts where that 4th point should appear. Only one solution will match.

---

# Real-World Examples

*   **Checkerboard Calibration:** This is done once in the factory for your smartphone camera or by a roboticist before a mission. If you drop your camera and the lens shifts, you have to re-calibrate using Zhang's method.
*   **P3P in AR (Pokemon Go / IKEA App):** When you point your phone at the floor, the app looks for "feature points" it recognizes. It uses a version of P3P to figure out exactly where your phone is tilted so it can place a 3D chair on the floor accurately.
*   **Drones:** A drone landing on a pad with specific markers uses P3P to calculate its distance and angle to the pad to ensure a soft landing.

---

# "Gotchas"

**Q: Why don't we just use one image for calibration?**
*   **Answer:** One image only gives us 2 equations. The matrix $B$ (which leads us to $K$) has 6 unknowns. Mathematically, the system is "underdetermined"—there are infinite solutions. You need at least 3 views to lock down one single answer.

**Q: Why must the checkerboard be moved to different angles?**
*   **Answer:** If you just move the checkerboard back and forth (staying parallel to the camera), you aren't providing new geometric information about the focal length or principal point. You need **rotation** (tilt) to help the math distinguish between "zoom" and "distance."

**Q: What happens in P3P if the 3 points are in a straight line (collinear)?**
*   **Answer:** The math fails. P3P requires the points to form a triangle. If they are in a line, the "pyramid" collapses, and you can't calculate the camera's position.

**Q: Why use a checkerboard and not a circle grid?**
*   **Answer:** Checkerboard corners are sharp and mathematically easy to define. However, under heavy motion blur, circles are sometimes better because the "center of gravity" of a blurry circle is easier to find than a blurry corner.

**Q: In Zhang's method, what happens to the 3rd column of the Rotation matrix?**
*   **Answer:** It is discarded during the Homography calculation because $Z=0$. However, once we find $K$ and $H$, we can "reconstruct" that 3rd column because we know $r_3$ must be perpendicular to $r_1$ and $r_2$ (Cross Product!).