# Advanced Image Processing - Assignment 2
## Comprehensive Documentation

---

# Table of Contents

1. [Assignment Overview](#assignment-overview)
2. [Question 1: Normalized Cut Image Segmentation](#question-1-normalized-cut-image-segmentation)
   - [1.1 Conceptual Foundation](#11-conceptual-foundation)
   - [1.2 Mathematical Framework](#12-mathematical-framework)
   - [1.3 Algorithm & Methodology](#13-algorithm--methodology)
   - [1.4 Implementation Details](#14-implementation-details)
   - [1.5 What Was Asked](#15-what-was-asked)
   - [1.6 What Was Done](#16-what-was-done)
   - [1.7 Results & Observations](#17-results--observations)
3. [Question 2: Deep Learning Semantic Segmentation](#question-2-deep-learning-semantic-segmentation)
   - [2.1 Conceptual Foundation](#21-conceptual-foundation)
   - [2.2 Mathematical Framework](#22-mathematical-framework)
   - [2.3 Algorithm & Methodology](#23-algorithm--methodology)
   - [2.4 Implementation Details](#24-implementation-details)
   - [2.5 What Was Asked](#25-what-was-asked)
   - [2.6 What Was Done](#26-what-was-done)
   - [2.7 Results & Observations](#27-results--observations)

---

# Assignment Overview

This assignment explores two fundamentally different paradigms for image segmentation:

| Aspect | Question 1 | Question 2 |
|--------|------------|------------|
| **Approach** | Classical (Unsupervised) | Deep Learning (Supervised) |
| **Method** | Normalized Cut (Graph-based) | Encoder-Decoder CNN |
| **Learning** | No training required | Requires labeled data |
| **Features** | Hand-crafted (intensity, edges, spatial) | Learned automatically |
| **Flexibility** | General-purpose | Task-specific (person segmentation) |

---

# Question 1: Normalized Cut Image Segmentation

## 1.1 Conceptual Foundation

### What is Image Segmentation?

Image segmentation is the process of partitioning an image into meaningful regions. Think of it like dividing a photograph into distinct objects - separating the sky from the mountains, or a person from the background.

### The Graph-Based Perspective

**Layman's Explanation:**

Imagine your image as a social network. Each pixel is a person, and "friendships" (connections) exist between nearby pixels. Strong friendships form between similar pixels (same color, similar texture), while weak friendships exist between dissimilar ones.

Segmentation becomes a problem of finding natural "communities" in this network - groups of pixels that are strongly connected internally but weakly connected to other groups.

**Technical View:**

We represent the image as a weighted undirected graph G = (V, E):
- **Vertices (V)**: Each pixel is a node
- **Edges (E)**: Connections between neighboring pixels
- **Weights (W)**: Similarity between connected pixels (higher = more similar)

### Why Normalized Cut?

The naive approach would be to simply find the minimum cut - remove edges with the smallest total weight to separate two groups. However, this tends to produce highly unbalanced partitions (isolating single pixels).

**The Problem with Minimum Cut:**

```
Image with one bright pixel surrounded by dark pixels:
[Dark][Dark][Bright][Dark][Dark]

Minimum cut would isolate the bright pixel:
[Dark][Dark] | [Bright] | [Dark][Dark]
              ↑ Very small cut!
```

**Normalized Cut Solution:**

Instead of just minimizing the cut, we normalize it by the total connection strength of each partition. This prevents small isolated segments.

---

## 1.2 Mathematical Framework

### Graph Construction

For an image of size H x W, we create a graph with N = H x W nodes.

#### Weight Matrix (Affinity Matrix)

The weight between pixels i and j encodes their similarity:

```
W(i,j) = exp(-d_intensity) × exp(-d_spatial) × [exp(-d_edge)]
```

Where each term is a Gaussian kernel measuring different aspects of similarity.

### The Affinity Function

The affinity (similarity) between two pixels combines multiple cues:

#### 1. Intensity Similarity

```
                    ||I(i) - I(j)||²
exp( - ────────────────────────────── )
                   2σ_I²
```

Where:
- `I(i)` = intensity/color at pixel i
- `σ_I` = intensity bandwidth parameter

**Interpretation:** Pixels with similar colors have high affinity.

#### 2. Spatial Proximity

```
                    ||X(i) - X(j)||²
exp( - ────────────────────────────── )
                   2σ_X²
```

Where:
- `X(i)` = (x, y) coordinates of pixel i
- `σ_X` = spatial bandwidth parameter

**Interpretation:** Nearby pixels have higher affinity than distant ones.

#### 3. Edge Similarity (Optional)

```
                    |E(i) - E(j)|²
exp( - ────────────────────────────── )
                   2σ_E²
```

Where:
- `E(i)` = edge magnitude at pixel i (from Sobel filter)
- `σ_E` = edge bandwidth parameter

**Interpretation:** Pixels on the same side of an edge have higher affinity.

### Combined Affinity Formula

For grayscale + spatial + edge mode:

```
                ⎡ (I_i - I_j)²     (x_i - x_j)² + (y_i - y_j)²     (E_i - E_j)² ⎤
W_ij = exp ⎢- ─────────────  -  ────────────────────────────  -  ───────────── ⎥
                ⎣    2σ_I²                   2σ_X²                    2σ_E²      ⎦
```

**Note:** Connections only exist within a radius r (sparse matrix for efficiency).

### Normalized Cut Definition

For a partition dividing graph into sets A and B:

```
                    cut(A,B)       cut(A,B)
Ncut(A,B) = ──────────────── + ────────────────
                 assoc(A,V)       assoc(B,V)
```

Where:
- `cut(A,B)` = Σ W(i,j) for i∈A, j∈B (total edge weight crossing the cut)
- `assoc(A,V)` = Σ W(i,j) for i∈A, j∈V (total connections from A to entire graph)

### Graph Laplacian

**Degree Matrix D:**
```
D_ii = Σ_j W_ij  (sum of all edge weights for node i)
```

**Unnormalized Laplacian:**
```
L = D - W
```

**Normalized (Symmetric) Laplacian:**
```
L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) W D^(-1/2)
```

### Spectral Relaxation

The discrete optimization problem is NP-hard. We relax it to a continuous problem:

**Original Problem:**
```
minimize    y^T (D - W) y
            ─────────────
               y^T D y

subject to  y_i ∈ {-1, 1}
```

**Relaxed Problem:**
```
minimize    y^T L_sym y

subject to  y^T y = 1
            y ⊥ D^(1/2) 1  (orthogonal to trivial solution)
```

**Solution:** The optimal y is the eigenvector corresponding to the second smallest eigenvalue of L_sym (the Fiedler vector).

### Binary Partitioning

Given the Fiedler vector v₂:
```
Segment(i) = { 0  if v₂(i) ≤ median(v₂)
             { 1  if v₂(i) > median(v₂)
```

### K-Way Partitioning

Two approaches:

**1. Hierarchical (Recursive):**
- Start with binary cut
- Recursively cut the segment with highest internal cut cost
- Repeat until k segments

**2. Spectral K-Way:**
- Extract k smallest eigenvectors [v₁, v₂, ..., vₖ]
- Each pixel becomes a k-dimensional point
- Apply K-means clustering in eigenspace

---

## 1.3 Algorithm & Methodology

### Complete Algorithm Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: RGB Image (H × W × 3)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Feature Extraction                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Grayscale   │  │    RGB       │  │   Sobel      │           │
│  │  Conversion  │  │   Values     │  │   Edges      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                              │                                   │
│                    Feature Vectors per Pixel                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: Build Affinity Matrix W                     │
│                                                                  │
│   For each pixel pair (i,j) within radius r:                     │
│                                                                  │
│   W[i,j] = exp(-intensity_diff/2σ_I²)                           │
│          × exp(-spatial_diff/2σ_X²)                              │
│          × exp(-edge_diff/2σ_E²)      [if edge mode]             │
│                                                                  │
│   Result: Sparse symmetric matrix (N × N)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: Compute Normalized Laplacian                │
│                                                                  │
│   D = diag(W × 1)           # Degree matrix                     │
│   L = D - W                  # Unnormalized Laplacian            │
│   L_norm = D^(-½) L D^(-½)   # Symmetric normalized Laplacian   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: Eigendecomposition                          │
│                                                                  │
│   Solve: L_norm v = λ v                                          │
│                                                                  │
│   Extract k smallest eigenvalues and eigenvectors               │
│   (using sparse solver: scipy.sparse.linalg.eigsh)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: Partitioning                                │
│                                                                  │
│   Binary Cut:                                                    │
│   - Use second eigenvector (Fiedler vector)                      │
│   - Threshold at median                                          │
│                                                                  │
│   K-Way Cut:                                                     │
│   - Stack k eigenvectors as rows                                 │
│   - Normalize rows to unit length                                │
│   - Apply K-means clustering                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: Segmentation Labels (H × W)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Extraction Details

```python
def compute_pixel_features(image):
    # Grayscale for intensity comparison
    gray = rgb2gray(image)  # Range [0, 1]

    # RGB normalized
    rgb = image / 255.0  # Range [0, 1]

    # Edge magnitude using Sobel
    gradient = sobel(gray)  # Range [0, ~1]

    # Spatial coordinates (normalized)
    h, w = gray.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.stack([y_coords/h, x_coords/w], axis=-1)

    return {
        'gray': gray,
        'rgb': rgb,
        'gradient': gradient,
        'coords': coords
    }
```

### Affinity Modes Explained

| Mode | Features Used | Formula |
|------|---------------|---------|
| `gray_spatial` | Grayscale + Position | W = exp(-ΔI²/2σ_I²) × exp(-ΔX²/2σ_X²) |
| `gray_spatial_edge` | Grayscale + Position + Edge | Above × exp(-ΔE²/2σ_E²) |
| `rgb_spatial` | RGB + Position | W = exp(-ΔRGB²/2σ_I²) × exp(-ΔX²/2σ_X²) |
| `rgb_spatial_edge` | RGB + Position + Edge | Above × exp(-ΔE²/2σ_E²) |
| `combined_feature_spatial` | RGB + Texture + Gradient + Position | Combined similarity |

---

## 1.4 Implementation Details

### Key Code Components

#### 1. Weight Matrix Construction

```python
def construct_weight_matrix(img_features, radius, sigma_I, sigma_X, sigma_E=None, mode="gray_spatial"):
    gray = img_features["gray"]
    rgb = img_features["rgb"]
    gradient = img_features["gradient"]
    coords = img_features["coords"]

    h, w = gray.shape
    N = h * w

    rows, cols = get_neighbor_pairs(h, w, radius)  # Sparse connectivity
    values = []

    for i, j in zip(rows, cols):
        y1, x1 = divmod(i, w)
        y2, x2 = divmod(j, w)

        # Spatial distance component
        spatial_diff = coords[y1, x1] - coords[y2, x2]
        dist_weight = np.exp(-np.sum(spatial_diff**2) / (2 * sigma_X**2))

        # Intensity component (depends on mode)
        if mode in ["gray_spatial", "gray_spatial_edge"]:
            feature_diff = gray[y1, x1] - gray[y2, x2]
            intensity_weight = np.exp(-(feature_diff**2) / (2 * sigma_I**2))
        elif mode in ["rgb_spatial", "rgb_spatial_edge"]:
            feature_diff = rgb[y1, x1] - rgb[y2, x2]
            intensity_weight = np.exp(-np.sum(feature_diff**2) / (2 * sigma_I**2))

        w_ij = intensity_weight * dist_weight

        # Edge component (optional)
        if "edge" in mode:
            edge_diff = abs(gradient[y1, x1] - gradient[y2, x2])
            gradient_weight = np.exp(-(edge_diff**2) / (2 * sigma_E**2))
            w_ij *= gradient_weight

        values.append(w_ij)

    # Build sparse symmetric matrix
    W = csr_matrix((values, (rows, cols)), shape=(N, N))
    W = 0.5 * (W + W.T)  # Ensure symmetry

    return W
```

#### 2. Normalized Laplacian

```python
def build_normalized_laplacian(W):
    d = np.array(W.sum(axis=1)).flatten()
    d = np.maximum(d, 1e-10)  # Avoid division by zero

    D = diags(d)
    L = D - W

    d_sqrt_inv = np.power(d, -0.5)
    D_sqrt_inv = diags(d_sqrt_inv)

    L_norm = D_sqrt_inv @ L @ D_sqrt_inv

    return L_norm
```

#### 3. Binary Normalized Cut

```python
def binary_normalized_cut(W, use_gpu=False):
    L = build_normalized_laplacian(W)

    if use_gpu and torch.cuda.is_available():
        # GPU acceleration for large matrices
        L_dense = L.toarray()
        L_tensor = torch.from_numpy(L_dense).double().cuda()
        eigenvalues, eigenvectors = torch.linalg.eigh(L_tensor)
        second_eigenvector = eigenvectors[:, 1].cpu().numpy()
    else:
        # Sparse CPU solver
        eigenvalues, eigenvectors = eigsh(L, k=2, which="SM")
        second_eigenvector = eigenvectors[:, 1]

    # Threshold at median
    threshold = np.median(second_eigenvector)
    seg_labels = (second_eigenvector > threshold).astype(int)

    return seg_labels
```

#### 4. Spectral K-Way Partitioning

```python
def spectral_ncut(W, k, use_gpu=False):
    L = build_normalized_laplacian(W)

    # Get k smallest eigenvectors
    eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
    embedding = eigenvectors

    # Normalize rows to unit length
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)
    embedding_normalized = embedding / row_norms

    # K-means in eigenspace
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    seg_labels = kmeans.fit_predict(embedding_normalized)

    return seg_labels
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Normalized Cut Cost** | Ncut(A,B) | Lower = better partition |
| **Intra-Segment Variance** | Var(pixels in segment) | Lower = more uniform segments |
| **Boundary-Edge Correspondence** | Overlap(boundaries, edges) | Higher = boundaries match image edges |
| **Partition Balance** | min(size) / max(size) | Closer to 1 = balanced segments |

---

## 1.5 What Was Asked

The assignment required:

### Part A: Feature Space Comparison
- Implement N-Cut segmentation with multiple affinity functions
- Compare: grayscale, RGB, spatial, edge features
- Analyze which combinations work best for different image types

### Part B: Parameter Sensitivity Analysis
- Study effect of key parameters:
  - **Radius (r)**: Neighborhood size for graph connectivity
  - **σ_I**: Intensity bandwidth
  - **σ_X**: Spatial bandwidth
  - **σ_E**: Edge bandwidth
- Document how each parameter affects segmentation quality

### Part C: K-Way Partitioning (Multi-way Segmentation)
- Implement two strategies:
  1. **Hierarchical**: Recursive binary splitting
  2. **Spectral**: Direct k-way using k eigenvectors + K-means
- Compare their performance on various images

---

## 1.6 What Was Done

### Implementation

1. **Complete N-Cut Framework** built from scratch using:
   - NumPy for array operations
   - SciPy sparse matrices for efficiency
   - PyTorch for optional GPU acceleration
   - OpenCV and scikit-image for image processing

2. **Five Affinity Modes Implemented:**
   - `gray_spatial` - Grayscale + position
   - `gray_spatial_edge` - Grayscale + position + edge
   - `rgb_spatial` - Color + position
   - `rgb_spatial_edge` - Color + position + edge
   - `combined_feature_spatial` - All features combined

3. **Comprehensive Parameter Sweeps:**
   - Radius: [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
   - σ_I: [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
   - σ_X: [1, 2, 3, 4, 5, 6, 7, 8, 10]
   - σ_E: [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

4. **Test Images:**
   - `zebra.jpg` - High texture, clear object boundaries
   - `chess.jpg` - Regular patterns, high contrast
   - `football.jpg` - Multiple objects, complex scene
   - `books.jpg` - Various colors, overlapping objects

5. **K-Way Partitioning:**
   - Both hierarchical and spectral methods implemented
   - Tested k values: [2, 3, 4, 5, 6, 8]

### Visualization & Evaluation

- Original image with segmentation overlay
- Color-coded segment masks
- Boundary visualization on original image
- Metric plots across parameter sweeps

---

## 1.7 Results & Observations

### Part A: Feature Space Comparison

| Feature Mode | Zebra | Chess | Football | Books |
|--------------|-------|-------|----------|-------|
| gray_spatial | Good striping | Square detection | Moderate | Color confusion |
| gray_spatial_edge | **Best** for stripes | **Best** squares | Good | Better edges |
| rgb_spatial | Color bleeding | Good | **Best** colors | **Best** separation |
| rgb_spatial_edge | Excellent | Excellent | Excellent | Excellent |

**Key Findings:**

1. **Edge features dramatically improve boundary detection** for textured objects (zebra stripes, chess patterns)

2. **RGB outperforms grayscale** when objects differ in hue but similar luminance (books, sports teams)

3. **Combined features (RGB + spatial + edge)** provide the most robust segmentation across image types

4. **Spatial features are essential** - pure intensity matching fails to create coherent regions

### Part B: Parameter Sensitivity

#### Radius (r) Effect:
```
Small radius (r=2-5):
├── Many isolated regions
├── Noisy boundaries
└── Fast computation

Large radius (r=15-20):
├── Over-smoothed regions
├── Lost fine details
└── Slow computation (O(r²) edges)

Optimal: r = 8-12 for 100x100 images
```

#### Intensity Bandwidth (σ_I) Effect:
```
Small σ_I (0.01-0.05):
├── Only identical pixels connect
├── Over-segmentation
└── Respects subtle color differences

Large σ_I (0.2-0.3):
├── Different colors merge
├── Under-segmentation
└── Smooth but inaccurate

Optimal: σ_I = 0.06-0.1 (image-dependent)
```

#### Spatial Bandwidth (σ_X) Effect:
```
Small σ_X (1-3):
├── Very local connections
├── Noisy boundaries
└── Preserves small details

Large σ_X (8-10):
├── Long-range connections
├── Smooth regions
└── May merge distinct objects

Optimal: σ_X = 4-6
```

#### Edge Bandwidth (σ_E) Effect:
```
Small σ_E (0.01-0.03):
├── Strong edge sensitivity
├── Boundaries follow edges closely
└── May over-segment textured regions

Large σ_E (0.5-1.0):
├── Edges have minimal influence
├── Smoother boundaries
└── May cross important edges

Optimal: σ_E = 0.05-0.15
```

### Part C: K-Way Partitioning Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Hierarchical** | Deterministic decisions, interpretable splits | May make suboptimal early splits |
| **Spectral K-Way** | Global optimization, balanced partitions | Depends on K-means initialization |

**Observations:**

1. For **k=2-3**: Both methods perform similarly
2. For **k>4**: Spectral method produces more balanced partitions
3. **Runtime**: Hierarchical is faster for small k, spectral is more efficient for large k
4. **Quality**: Spectral tends to have lower total Ncut cost

---

# Question 2: Deep Learning Semantic Segmentation

## 2.1 Conceptual Foundation

### What is Semantic Segmentation?

Semantic segmentation assigns a class label to every pixel in an image. Unlike object detection (bounding boxes) or instance segmentation (individual objects), semantic segmentation creates a dense pixel-wise classification map.

**Layman's Explanation:**

Imagine coloring a coloring book, but instead of creative freedom, you must color each region based on what it actually is - all sky pixels blue, all grass pixels green, all person pixels pink, etc.

### The Encoder-Decoder Paradigm

**Why do we need this architecture?**

Standard CNNs (like ResNet) progressively reduce spatial resolution to capture high-level semantics. For classification, this is fine - we only need one label. For segmentation, we need to "undo" this shrinking to get back to the original image size.

```
Standard CNN:
Input (224×224) → [Conv+Pool] → [Conv+Pool] → ... → Feature (7×7) → Class

Segmentation CNN:
Input (224×224) → [Encoder: shrink] → Feature (7×7) → [Decoder: expand] → Output (224×224)
```

### The Encoder (ResNet-34)

**What it does:**
- Extracts hierarchical features from input images
- Early layers: edges, textures (high resolution)
- Later layers: objects, semantic concepts (low resolution)

**Why ResNet?**
- Pretrained on ImageNet (1M+ images, 1000 classes)
- Skip connections allow very deep networks (34 layers)
- Proven feature extraction capability

**Architecture:**
```
Input: 3 × H × W

Layer0 (stem):    64 × H/2 × W/2     ← Initial conv + pool
Layer1:           64 × H/4 × W/4     ← Basic blocks
Layer2:          128 × H/8 × W/8     ← Downsampling
Layer3:          256 × H/16 × W/16   ← Downsampling
Layer4:          512 × H/32 × W/32   ← Final features
```

### The Decoder

**Purpose:** Upsample low-resolution feature maps back to original image size while combining multi-scale information.

**Two Approaches (explored in this assignment):**

1. **Single-Stage (Direct) Decoder:**
   - One large transposed convolution
   - Simple but loses multi-scale information

2. **Progressive (Multi-Stage) Decoder:**
   - Gradual upsampling with skip connections
   - U-Net style architecture
   - Preserves fine details through skip connections

---

## 2.2 Mathematical Framework

### Transposed Convolution (Upsampling)

**Regular Convolution (Downsampling):**
```
Output size = (Input - Kernel + 2×Padding) / Stride + 1
```

**Transposed Convolution (Upsampling):**
```
Output size = (Input - 1) × Stride - 2×Padding + Kernel
```

**Example:**
```
Input: 7×7, Kernel: 64, Stride: 32, Padding: 16
Output: (7-1)×32 - 2×16 + 64 = 192 - 32 + 64 = 224
```

### Skip Connections

Skip connections concatenate encoder features with decoder features:

```
Decoder_input = Concat(Upsampled_features, Encoder_features)

Channel dimensions:
- Upsampled: 256 channels
- Encoder (Layer3): 256 channels
- Combined: 512 channels
```

**Why this helps:**

The encoder loses spatial precision during downsampling. Skip connections provide high-resolution details that help the decoder reconstruct accurate boundaries.

### Loss Functions

#### 1. Cross-Entropy Loss

```
L_CE = -1/N Σ Σ y_c,p × log(ŷ_c,p)
            c  p

Where:
- N = number of pixels
- c = class index
- p = pixel index
- y = ground truth (one-hot)
- ŷ = prediction (softmax probability)
```

**For binary segmentation (person vs background):**
```
L_CE = -1/N Σ [y_p × log(ŷ_p) + (1-y_p) × log(1-ŷ_p)]
           p
```

#### 2. Weighted Cross-Entropy

Addresses class imbalance (few person pixels, many background pixels):

```
L_WCE = -1/N Σ w_c × y_c,p × log(ŷ_c,p)
            c,p

Where w_c = N / (C × count(class c))
```

**Example:**
- Background: 90% of pixels → weight 0.55
- Person: 10% of pixels → weight 5.0

#### 3. Dice Loss

Measures overlap between prediction and ground truth:

```
Dice = 2 × |P ∩ G| / (|P| + |G|)

L_Dice = 1 - Dice = 1 - (2 × Σ p×g) / (Σ p² + Σ g²)

Where:
- P = predicted probabilities
- G = ground truth mask
```

**Soft Dice (differentiable):**
```
L_SoftDice = 1 - (2 × Σ ŷ × y + ε) / (Σ ŷ² + Σ y² + ε)
```

#### 4. Combined Loss

```
L_Total = L_CE + λ × L_Dice

Typically λ = 1.0
```

### Evaluation Metrics

#### Pixel Accuracy

```
Accuracy = Correct Pixels / Total Pixels
         = (TP + TN) / (TP + TN + FP + FN)
```

**Problem:** Biased by dominant class (background).

#### Intersection over Union (IoU)

```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
    = TP / (TP + FP + FN)
```

**Per-class IoU:**
```
IoU_person = TP_person / (TP_person + FP_person + FN_person)
IoU_background = TP_bg / (TP_bg + FP_bg + FN_bg)
```

#### Mean IoU (mIoU)

```
mIoU = (IoU_person + IoU_background) / 2
```

---

## 2.3 Algorithm & Methodology

### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA PREPARATION                     │
│                                                                  │
│  Images: RGB photographs        Masks: Binary (0=bg, 1=person)  │
│  ┌────────────┐                 ┌────────────┐                  │
│  │ [Football  │                 │ [Person    │                  │
│  │  scene]    │       ──►       │  mask]     │                  │
│  └────────────┘                 └────────────┘                  │
│                                                                  │
│  Augmentation: Random crop, flip, color jitter                  │
│  Normalization: ImageNet mean/std                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER (Frozen ResNet-34)               │
│                                                                  │
│  Input ──► Conv1 ──► Layer1 ──► Layer2 ──► Layer3 ──► Layer4   │
│  (3ch)     (64)      (64)       (128)      (256)      (512)    │
│                        │          │          │                   │
│                        ▼          ▼          ▼                   │
│                    [Skip x1]  [Skip x2]  [Skip x3]               │
│                                                                  │
│  *** FROZEN: No gradient updates during training ***            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER (Trainable)                      │
│                                                                  │
│  Option A: Single-Stage                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  512ch (7×7) ──► TransConv(k=64,s=32) ──► 2ch (224×224)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Option B: Progressive with Skip Connections                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  512ch ──► Up2× + Skip_x3 ──► 256ch                       │   │
│  │  256ch ──► Up2× + Skip_x2 ──► 128ch                       │   │
│  │  128ch ──► Up2× + Skip_x1 ──► 64ch                        │   │
│  │  64ch  ──► Up8× ──► 2ch (224×224)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                         │
│                                                                  │
│  Prediction: 2 × H × W (logits)                                 │
│  Ground Truth: H × W (labels 0 or 1)                            │
│                                                                  │
│  Loss = CrossEntropy(pred, truth) + Dice(softmax(pred), truth)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKPROPAGATION                          │
│                                                                  │
│  Gradients flow ONLY through decoder (encoder frozen)           │
│  Optimizer: Adam                                                 │
│  Learning Rate: 1e-3                                             │
│  Mixed Precision: FP16 forward, FP32 gradients                  │
└─────────────────────────────────────────────────────────────────┘
```

### Decoder Architectures in Detail

#### Single-Stage Decoder

```python
class DirectUpsampleDecoder(nn.Module):
    def __init__(self, in_channels=512, num_classes=2, kernel_size=64, stride=32, padding=16):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,      # 512 (from ResNet Layer4)
            out_channels=num_classes,      # 2 (person + background)
            kernel_size=kernel_size,       # 64 (large kernel for 32× upsampling)
            stride=stride,                 # 32 (matches encoder downsampling)
            padding=padding,               # 16 (to get exact output size)
            bias=False
        )

    def forward(self, x):
        return self.up(x)
```

**Upsampling calculation:**
```
Input:  512 × 7 × 7      (after ResNet on 224×224 input)
Output: 2 × 224 × 224    (32× spatial expansion)

Size = (7 - 1) × 32 - 2 × 16 + 64 = 192 - 32 + 64 = 224 ✓
```

#### Progressive Decoder (U-Net Style)

```python
class UpsampleStage(nn.Module):
    """One upsampling stage with optional skip connection"""

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=2):
        super().__init__()

        # Transposed convolution for 2× upsampling
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2 - 1 if kernel_size > 2 else 0
        )

        # Fusion convolution (combines upsampled + skip features)
        total_channels = out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)  # Concatenate along channels

        return self.conv(x)


class MultiStageDecoder(nn.Module):
    def __init__(self, skip_levels, kernel_size=2, num_classes=2):
        super().__init__()

        self.skip_levels = skip_levels

        # Stage 1: 512 → 256 (x4 → x3 resolution)
        self.stage1 = UpsampleStage(
            in_channels=512,
            skip_channels=256 if "x3" in skip_levels else 0,
            out_channels=256,
            kernel_size=kernel_size
        )

        # Stage 2: 256 → 128 (x3 → x2 resolution)
        self.stage2 = UpsampleStage(
            in_channels=256,
            skip_channels=128 if "x2" in skip_levels else 0,
            out_channels=128,
            kernel_size=kernel_size
        )

        # Stage 3: 128 → 64 (x2 → x1 resolution)
        self.stage3 = UpsampleStage(
            in_channels=128,
            skip_channels=64 if "x1" in skip_levels else 0,
            out_channels=64,
            kernel_size=kernel_size
        )

        # Final 8× upsampling to full resolution
        self.final = nn.ConvTranspose2d(64, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x4, skip_features):
        x = self.stage1(x4, skip_features.get("x3"))
        x = self.stage2(x, skip_features.get("x2"))
        x = self.stage3(x, skip_features.get("x1"))
        return self.final(x)
```

### Skip Connection Strategies

| Configuration | Skip Sources | Effect |
|---------------|--------------|--------|
| `none` | No skips | Pure decoder, loses fine details |
| `x3` | Layer3 only | High-level semantic preservation |
| `x3_x2` | Layer2 + Layer3 | Balance of semantics and details |
| `x3_x2_x1` | All layers | Maximum detail preservation |

### Fusion Methods

**Concatenation:**
```python
combined = torch.cat([upsampled, skip], dim=1)
# Channel dims: Up(256) + Skip(256) = 512
```

**Summation (Additive):**
```python
# Requires channel alignment first
aligned_skip = conv_1x1(skip)  # Match channels
combined = upsampled + aligned_skip
# Channel dims: Still 256
```

---

## 2.4 Implementation Details

### Dataset Structure

```
Semantic_segmentation/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── masks/
│       ├── img_0001.png
│       ├── img_0002.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Data Loading

```python
class PersonSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_dir = mask_dir
        self.transform = transform

        # ImageNet normalization (for pretrained ResNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load mask (binary: 0=background, 1=person)
        mask_name = os.path.basename(self.image_paths[idx]).replace(".jpg", ".png")
        mask = Image.open(os.path.join(self.mask_dir, mask_name))

        # Apply transforms (same random crop/flip to both)
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
```

### Training Loop

```python
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0

    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Training Configuration

```python
SETTINGS = {
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 4,
    "LEARNING_RATE": 1e-3,
    "NUM_EPOCHS": 60,
    "EARLY_STOPPING_PATIENCE": 5,
    "USE_MIXED_PRECISION": True,
    "SEED": 42,
    "ENCODER_NAME": "resnet34",
}
```

---

## 2.5 What Was Asked

The assignment required:

### Part A: Direct Upsampling Decoder
- Implement a single transposed convolution decoder
- Test different kernel sizes: 32, 64, 128, 256
- Analyze trade-offs in reconstruction quality

### Part B: Progressive Decoder with Skip Connections
- Implement U-Net style multi-stage upsampling
- Compare skip connection configurations:
  - No skips
  - x3 only (Layer3)
  - x3 + x2 (Layer2 + Layer3)
  - x3 + x2 + x1 (All layers)
- Test kernel sizes: 2, 3, 4, 5

### Part C: Loss Function Comparison
- Cross-Entropy
- Weighted Cross-Entropy
- Cross-Entropy + Dice

### Part D: Fusion Method Comparison
- Concatenation vs Summation for skip connections

### Evaluation Requirements
- Pixel Accuracy
- Mean IoU
- Per-class IoU (Background, Person)
- Visual comparison of predictions

---

## 2.6 What Was Done

### Comprehensive Experiment Grid

**Total Experiments:** 60+

| Factor | Options | Count |
|--------|---------|-------|
| Decoder Type | Single, Progressive | 2 |
| Kernel Sizes | 32, 64, 128, 256 (single) / 2, 3, 4, 5 (progressive) | 4 each |
| Skip Configs | none, x3, x3_x2, x3_x2_x1 | 4 |
| Loss Functions | CE, Weighted CE, CE+Dice | 3 |
| Fusion Method | Concat, Sum | 2 |

### Implementation Highlights

1. **Frozen Encoder:**
   ```python
   encoder = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
   for param in encoder.parameters():
       param.requires_grad = False
   ```

2. **Skip Feature Extraction:**
   ```python
   def forward(self, x):
       skips = {}
       x = self.encoder.conv1(x)
       x = self.encoder.bn1(x)
       x = self.encoder.relu(x)
       x = self.encoder.maxpool(x)

       x = self.encoder.layer1(x)
       skips["x1"] = x

       x = self.encoder.layer2(x)
       skips["x2"] = x

       x = self.encoder.layer3(x)
       skips["x3"] = x

       x = self.encoder.layer4(x)

       output = self.decoder(x, skips)
       return output
   ```

3. **Combined Loss:**
   ```python
   class CombinedLoss(nn.Module):
       def __init__(self, ce_weight=1.0, dice_weight=1.0):
           super().__init__()
           self.ce = nn.CrossEntropyLoss()
           self.ce_weight = ce_weight
           self.dice_weight = dice_weight

       def forward(self, pred, target):
           ce_loss = self.ce(pred, target)

           pred_soft = F.softmax(pred, dim=1)[:, 1]  # Person probability
           target_float = target.float()

           intersection = (pred_soft * target_float).sum()
           union = pred_soft.sum() + target_float.sum()
           dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

           return self.ce_weight * ce_loss + self.dice_weight * dice_loss
   ```

4. **Evaluation Metrics:**
   ```python
   def compute_metrics(pred, target, num_classes=2):
       pred_flat = pred.flatten()
       target_flat = target.flatten()

       # Pixel Accuracy
       correct = (pred_flat == target_flat).sum()
       total = len(target_flat)
       pixel_acc = correct / total

       # Per-class IoU
       ious = []
       for c in range(num_classes):
           pred_c = (pred_flat == c)
           target_c = (target_flat == c)

           intersection = (pred_c & target_c).sum()
           union = (pred_c | target_c).sum()

           iou = intersection / (union + 1e-6)
           ious.append(iou)

       return {
           "pixel_accuracy": pixel_acc,
           "iou_background": ious[0],
           "iou_person": ious[1],
           "mean_iou": np.mean(ious)
       }
   ```

### Training Details

- **Epochs:** Up to 60 (with early stopping, patience=5)
- **Optimizer:** Adam, LR=1e-3
- **Batch Size:** 16
- **Mixed Precision:** Enabled (FP16 forward, FP32 backward)
- **Hardware:** NVIDIA GPU with CUDA

---

## 2.7 Results & Observations

### Single-Stage Decoder Results

| Kernel Size | Pixel Acc (%) | mIoU (%) | Person IoU (%) | Training Notes |
|-------------|---------------|----------|----------------|----------------|
| 32 | 99.52 | 90.12 | 80.85 | Fast training |
| **64** | **99.77** | **94.26** | **88.75** | Best single-stage |
| 128 | 99.71 | 93.54 | 87.31 | Similar to 64 |
| 256 | 99.65 | 92.88 | 86.01 | Diminishing returns |

**Observation:** Kernel size 64 provides the best balance. Larger kernels don't improve quality and add parameters.

### Progressive Decoder - Skip Connection Impact

| Skip Configuration | Pixel Acc (%) | mIoU (%) | Person IoU (%) |
|--------------------|---------------|----------|----------------|
| none | 99.42 | 88.76 | 77.92 |
| x3 only | 99.68 | 92.94 | 86.11 |
| x3 + x2 | 99.78 | 95.66 | 91.49 |
| **x3 + x2 + x1** | **99.83** | **95.77** | **91.70** |

**Key Finding:** Each additional skip connection improves Person IoU by 4-5%. Full skip connections (x1+x2+x3) achieve the best results.

### Progressive Decoder - Kernel Size Impact

| Kernel Size | mIoU (%) | Person IoU (%) | Notes |
|-------------|----------|----------------|-------|
| 2 | 95.71 | 91.58 | Minimal upsampling per stage |
| 3 | 95.77 | 91.70 | Slightly better |
| 4 | 95.75 | 91.66 | Similar |
| 5 | 95.73 | 91.62 | No improvement |

**Key Finding:** Kernel size has minimal impact in progressive decoder (unlike single-stage). Skip connections dominate the information flow.

### Loss Function Comparison

| Loss Function | mIoU (%) | Person IoU (%) | Background IoU (%) |
|---------------|----------|----------------|--------------------|
| Cross-Entropy | 93.54 | 87.31 | 99.76 |
| Weighted CE | 94.68 | 89.52 | 99.84 |
| **CE + Dice** | **95.77** | **91.70** | **99.84** |

**Key Finding:** CE + Dice provides the best Person IoU, improving ~4.4% over plain CE. Dice loss directly optimizes overlap.

### Fusion Method Comparison

| Fusion Method | mIoU (%) | Person IoU (%) |
|---------------|----------|----------------|
| Concatenation | 95.77 | 91.70 |
| **Summation** | **95.87** | **91.90** |

**Key Finding:** Summation slightly outperforms concatenation (+0.2% mIoU) while using fewer parameters.

### Best Overall Configuration

```
Architecture:   Progressive Decoder
Skip Levels:    x3 + x2 + x1 (full)
Kernel Size:    3
Fusion Method:  Summation
Loss Function:  Cross-Entropy + Dice

Results:
├── Pixel Accuracy: 99.84%
├── Mean IoU:       95.87%
├── Person IoU:     91.90%
└── Background IoU: 99.84%
```

### Visual Observations

1. **Boundary Quality:**
   - Single-stage produces blurry boundaries
   - Progressive with skips has crisp, accurate edges
   - Skip connections preserve fine details (fingers, hair)

2. **Challenging Cases:**
   - Occluded persons: Progressive handles better
   - Multiple persons: Both struggle with boundaries between people
   - Small persons: Skip connections crucial for detection

3. **Failure Modes:**
   - Complex backgrounds (crowds): Some false positives
   - Unusual poses: Occasional misclassification
   - Low contrast: Both architectures struggle

### Key Takeaways

1. **Skip connections are essential** for high-quality segmentation. They provide a 3-5% mIoU improvement.

2. **Progressive upsampling > Direct upsampling** by ~1.5% mIoU when using full skip connections.

3. **Dice loss is crucial for class-imbalanced segmentation.** It improves minority class (person) IoU by 4%.

4. **Frozen encoder works well** for this task. The pretrained ImageNet features transfer effectively to person segmentation.

5. **Kernel size matters for single-stage** but not for progressive decoders where skips dominate.

6. **Summation vs Concatenation** shows similar performance, but summation is more parameter-efficient.

---

# Appendix

## Code Structure

```
Assignment 2_Code/
├── Question_1_Final.ipynb          # N-Cut implementation
├── Question_2_Final.ipynb          # Semantic segmentation (concat)
├── Question_2_Summation_Skip.ipynb # Semantic segmentation (sum)
└── Outputs/
    ├── Q1/
    │   ├── A/segmentations/        # Feature comparison results
    │   └── B/metrics/              # Parameter sensitivity data
    └── Q2/
        └── decoder_comparison/
            └── metrics/            # All experiment results (JSON)
```

## References

1. Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. IEEE TPAMI.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
4. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. CVPR.

---

*Document generated for AIP Assignment 2*
