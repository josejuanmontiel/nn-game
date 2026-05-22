# 🌌 Deep Space Stretching: The Neural Manifold Encyclopedia
### A Tactile Field Guide to Deep Learning (NYU DS-GA 1008 / CILVR Lab)

Welcome to the definitive companion to the **Deep Space Stretching** simulator. This document is a synthesis of four foundational NYU Deep Learning transcripts, designed to turn the "black box" of neural networks into a geometric, biological, and historical landscape.

### 📚 Quick Links
- **[Documentation Index](doc/README.md)** (Architecture, Manifolds, Config)
- **[Task List (TODO)](TODO.md)**

---

## 🏛️ I. The Epochs of Artificial Intelligence

Our simulator is a window into a 70-year evolution of thought.

### 1. The Logical Foundations (1940s)
Before we had "Deep Learning," we had **Cybernetics**.
- **McCulloch & Pitts (1943)**: Proposed the first mathematical model of a neuron. They believed the brain was a **logical inference machine**. If neurons are binary switches (ON/OFF), they can represent AND/OR/NOT gates.
- **Donald Hebb (1949)**: Coined the rule of **Synaptic Plasticity**: *"Neurons that fire together, wire together."* This is the biological basis for increasing weight values between correlated stars in this game.
- **Norbert Wiener**: Father of Cybernetics, who taught us about **Feedback Loops**—the precursor to the error correction you see in the simulator's Step function.

### 2. The Analog Era (1950s-60s)
**Frank Rosenblatt** built the **Perceptron**, but it wasn't code. It was a room-sized machine with **motorized potentiometers** (physical knobs) that rotated to adjust weights.
- **Bernie Widrow** followed with the **ADALINE**, an even more compact analog learner.
- **The AI Winter (1969)**: Minsky and Papert proved that linear perceptrons couldn't solve the "Spiral" or "XOR" problems. Research died for 15 years because researchers hadn't yet embraced the **Continuous Neuron**.

### 3. The Backprop Summer (1980s - Present)
The "Calculus of Neural Networks" was born from a shift in economics.
- **The Cost of Multiplication**: Before 1985, floating-point multiplication was "ridiculously expensive." Binary neurons were used because they only required *addition*. 
- **The Modern Wave**: Deep Learning took over the world in three steps:
    - **2010**: Speech Recognition (Android).
    - **2012**: Computer Vision (The ImageNet/Convolutional revolution).
    - **2015**: NLP (Transformers/Translation).

---

## 🔄 II. Galactic Geometry: The Space Fabric

### 1. Linear Algebra as "Space Motion"
Every weight matrix ($W$) in this game is a **Geometric Transformer**.
*   **Rotation**: Spinning the data points to find a better orientation.
*   **Scaling (SVD)**: Every matrix has "Singular Values." If a singular value is 2.0, that layer **zooms** in that direction. If it's 0.1, it **squashes** it.
*   **Reflection**: If the "Determinant" of your weight matrix is negative, the space is flipped (mirrored).
*   **Translation ($b$)**: Shifting the entire universe so the camera centers on the data.

### 2. The "Aperture" (Hidden Layers)
Why not just one giant layer?
- **Width vs Depth**: A "fat" network is like a huge parallel processor. A "deep" network (stacking layers) allows for **Compositionality**.
- **Exponential Complexity**: Three layers of 10 neurons can represent the same complexity as one layer of 1,000 neurons, but with significantly less memory.

### 3. The Manifold Hypothesis
Real data (stars) live on a **low-dimensional manifold**. Imagine a piece of crumpled paper in a 3D room. 
- **The Unfolding**: The network's goal is to reach into the 3D space and pull the paper flat (2D). 
- **Tangents**: In high-D, points that look far apart are actually close on the surface of the manifold. Deep Learning is the art of **unshrinking distances**.

---

## 🧬 III. The Biological Blueprint (The Eye of the Squid)

The **Modo Auditoría** is a virtual electrode, mirroring the experiments of **Hubel & Wiesel**.

### 1. Evolution's Mistake (Vertebrate Vision)
As discussed in **Lecture 1**, the human eye is a 100-megapixel camera, but our optical nerve only has 1 million fibers. 
- **The Compression**: Retinal neurons perform lossy compression before sending signals to the brain.
- **The Blind Spot**: Human eyes have wires (nerves) in *front* of the sensors, punching a hole through the retina. 
- **The Ideal**: **Invertebrates (Squids/Octopi)** have wires in the *back*. They have no blind spot. Our simulator is a "Squid's Eye"—perfect, unblocked visibility into the data.

### 2. The Ventral Hierarchy
Your brain processes the "Space Fabric" in stages:
1.  **V1**: Simple edges.
2.  **V2/V4**: Textures and complex shapes.
3.  **IT Cortex**: The "Jennifer Aniston" neuron—cells that fire only for specific high-level concepts (like "Blue Spiral").

---

## ⛰️ IV. The Engine of Change: Learning

### 1. The Redundancy of the Universe
Why do we use **Stochastic Gradient Descent (SGD)** instead of Batch?
- **Data Duplication**: If you have 1,000,000 samples, most are duplicates. If you look at every point before taking a step, you are wasting 99% of your computation.
- **The "Vibe" of the Gradient**: One sample gives you the approximate direction of "downhill." It's like walking in a fog—you don't need a map of the whole mountain to know where the next step goes.

### 2. The "Blame" Calculus (Backprop)
The **Pizarra** mathematically visualizes the **Chain Rule**:
- **Output Layer**: Computes the "Logits" and applies the **Soft Argmax** (Softmax).
- **The Softmax Logic**: It converts raw numbers into probabilities. If one number is 10 and others are 1, Softmax makes the 10 nearly 100% and others 0%.
- **Gradient Flow**: Using the $f'(z)$ derivative, the network sends a "signal of blame" backwards. If a neuron was "wrong," its weights are "shrunk" (squashed) in the direction that minimizes cost.

---

## 🧪 V. Interactive Field Guide: Experiments

### 1. The Symmetry Breaking
Reset a level and notice weights start random. If all weights were the same (Zero or Identity), the network would never learn because every neuron would compute the same gradient. **Randomness is the fuel for diversity.**

### 2. The Vanishing Gradient
Stack 10 layers with **Tanh**. Notice how training becomes slow and the "Space Fabric" barely moves. This is the **Vanishing Gradient**: the math "dies" as derivatives multiply into tiny numbers. Switch to **ReLU** to "re-awaken" the deep layers.

### 3. The Generalization Gap
Watch the accuracy on Level 3. If it reaches 100% on the stars but the grid looks "jagged" and messily folded, you have **Overfitted**. A beautiful, smooth "Space Fabric" usually means the network has understood the universal rule of the spiral.

---

## 🚀 VI. New Galactic Tools: Interaction & Config

### 1. The Probe Launcher (Real-Time Inference)
The simulator now allows for **Active Probing**. 
- **Shift + Left Click**: Launches a "Probe" from the player's position to the mouse cursor.
- **Trace Visualization**: Watch the probe's trajectory as it is warped by the manifold in real-time.
- **Star Collapse**: Upon arrival, the probe "collapses" into a permanent star, classified according to the active level's pattern.
- **Ghost Target**: A 3D ghost cursor follows your mouse, showing exactly where your 2D position maps in the hidden manifold.

### 2. External Intelligence: `levels.json`
We have moved all level parameters to an external configuration file. You can now tune the universe without recompling:
- **Point Separation**: Adjust `RadiusScale` and `Separation` to change how "noisy" or "tight" the data clusters are.
- **Spiral Topology**: Control the number of `SpiralRotations` (twistiness) of the intertwining arms.
- **Initialization Energy**: Tune `WeightInitScale` and `BiasInitScale` to control the network's starting "stiffness."
- **Engine Selection**: Each level can now specify its own optimization engine via `"EngineType"`.
- **GPU Activation**: See the Building section below for how to enable GPU-accelerated engines.

### 3. Building & Running with GPU Support
By default, the simulator runs using Go's standard/parallel engines. To enable the hardware-accelerated **Level Zero** or **oneAPI** engines, you must compile with the `gpu` build tag:

```bash
# Standard build (CPU only)
go run .

# GPU-Accelerated build
go build -tags gpu -o game_gpu main.go
./game_gpu
```
> [!NOTE]
> GPU engines require the respective Intel drivers and SDKs (oneAPI/Level Zero) to be installed on your system.

### 4. Multiplexed Math Engines (`nn/` Structure)
The calculation logic is organized into specialized backends for maximum flexibility and educational value:
- `nn/standard/`: Sequential Go implementation (Stable).
- `nn/parallel/`: Multi-threaded Go (Goroutines).
- `nn/avx/`: Assembly-accelerated vector operations (x86_64).
- `nn/oneapi/`: GPU acceleration via Intel oneAPI/SYCL.
- `nn/level0/`: Direct low-level GPU control via Level Zero.

---

## 🎮 VII. Encyclopedia Keybindings
Modeled after the pedagogical philosophy of the NYU Deep Learning course.

| Legend | Concept | Simulator Action |
| :--- | :--- | :--- |
| **Shift+Click** | **Probe Launch** | Direct inference test with animated trajectory. |
| **T** | **Stochastic Ascent** | Engage the SGD weight update engine. |
| **S** | **Nodal Audit** | Step-by-step math evaluation on the Pizarra (Hover base of screen). |
| **P/M** | **Architectural Depth** | Increase/Decrease hierarchy (V1 -> IT). |
| **R/C/A** | **The Folding Tools** | ReLU (Sharp folds) / Tanh (Gliding folds) / Identity. |
| **Wheel** | **The Electrode** | "Poke" specific layers/biases/weights for inspection/adjustment. |
| **1-4 / N / J** | **The Manifolds** | Shift between increasingly entangled universes. |
| **levels.json** | **The Universe Source**| External file to tune difficulty, separation, and **EngineType**. |

---
*Dedicated to the pedagogical legacy of Yann LeCun & Alfredo Canziani.*
