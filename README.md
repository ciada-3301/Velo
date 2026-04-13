# Velo 🚀

**Real-time 3D Vision Navigation System with Intelligent Pathfinding**

Velo is a high-performance computer vision system that leverages Intel iGPU acceleration to perform real-time depth estimation, obstacle detection, floor plane extraction, and autonomous pathfinding from a standard webcam feed. Built for robotics, assistive navigation, and AR/VR applications.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.x-orange.svg)](https://github.com/openvinotoolkit/openvino)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Key Features

- **GPU-Accelerated Depth Estimation**: Utilizes Intel iGPU with OpenVINO runtime for real-time depth inference using Depth Anything V2/V3 models
- **Multi-Threaded Pipeline**: Optimized 4-stage pipeline (capture → inference → processing → visualization) for maximum throughput
- **Intelligent Floor Detection**: Histogram-based floor plane extraction with configurable tolerance
- **Robust Obstacle Detection**: Spatial binning algorithm for reliable obstacle identification above floor level
- **A* Pathfinding**: Real-time safe path planning around detected obstacles
- **Spline Path Smoothing**: B-spline interpolation for natural, robot-friendly trajectories
- **3D Visualization**: Interactive Rerun-based visualization with point clouds, depth maps, and path overlays

## 🏗️ System Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Camera     │─────▶│   GPU        │─────▶│  CPU         │─────▶│   Rerun      │
│  Capture     │      │  Inference   │      │  Processing  │      │  Logging     │
│  Thread      │      │  Thread      │      │  Thread      │      │  Thread      │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
     640×480              Depth Map          Floor/Obstacles        3D Viz + Path
    RGB Frames           (518×518)           A* Pathfinding         Overlay
```

### Pipeline Stages

1. **Camera Thread**: Captures frames at maximum camera framerate, drops old frames if inference lags
2. **GPU Inference Thread**: Runs Depth Anything V2 model on Intel iGPU via OpenVINO
3. **CPU Processing Thread**: 
   - Back-projects depth to 3D point cloud
   - Detects floor plane using Y-axis histogram
   - Identifies obstacles using spatial XZ-binning
   - Builds occupancy grid and computes A* path
   - Smooths path with cubic B-splines
4. **Rerun Logger Thread**: Streams visualization data without blocking computation

## 🛠️ Technical Details

### Depth Estimation
- **Model**: Depth Anything V2 (ViT-B) / V3 (Small) 
- **Input**: 518×518 RGB (normalized)
- **Output**: Metric depth map (0.3m - 4.0m range)
- **Inference Backend**: OpenVINO GPU runtime

### Floor Detection Algorithm
```python
# Histogram-based floor plane extraction
Y_BINS = 200                    # Vertical resolution
FLOOR_SLAB_M = 0.08             # ±8cm tolerance around floor Y
```
- Computes Y-axis histogram of 3D points
- Identifies peak bin (highest point density) as floor level
- Extracts all points within ±8cm slab

### Obstacle Detection
```python
XZ_BIN_SIZE = 0.05              # 5cm grid cells
MIN_OBSTACLE_POINTS = 3          # Minimum points per cell
OBSTACLE_SLAB_M = 0.10          # 10cm per vertical layer
OBSTACLE_SLABS = 2              # Check 20cm above floor
```
- Bins 3D space into 5cm×5cm XZ grid cells
- Counts points per cell in obstacle zone (0-20cm above floor)
- Cells with ≥3 points flagged as obstacles
- Vectorized NumPy operations (no Python loops)

### Pathfinding
```python
GRID_RESOLUTION = 0.1           # 10cm occupancy grid cells
GOAL_DISTANCE = 3.0             # Target: 3m ahead
```
- **Occupancy Grid**: 0 = free, 1 = obstacle, 2 = unknown
- **Algorithm**: A* with diagonal movement (8-way connectivity)
- **Heuristic**: Euclidean distance to goal
- **Safety Margin**: 2-iteration dilation on obstacles
- **Goal Snapping**: Automatically finds nearest free cell if goal blocked
- **Smoothing**: Cubic B-spline with `s=len(path)×0.6` relaxation factor

## 📋 Requirements

### Hardware
- **Camera**: Any USB/built-in webcam (640×480 minimum)
- **GPU**: Intel integrated GPU (Gen9+) or discrete GPU
- **CPU**: Multi-core recommended (4+ threads)
- **RAM**: 4GB minimum, 8GB recommended

### Software Dependencies

```bash
# Core dependencies
opencv-python>=4.8.0
openvino>=2024.0
numpy>=1.24.0
scipy>=1.11.0
rerun-sdk>=0.17.0

# Model conversion (optional)
torch>=2.0.0
depth-anything-v3  # For model export
```

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/ciada-3301/Velo.git
cd Velo
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install opencv-python openvino numpy scipy rerun-sdk
```

### 4. Download/Convert Model

#### Option A: Use Pre-converted OpenVINO Model
1. Download `depth_anything_v2_vitb.xml` and `depth_anything_v2_vitb.bin`
2. Place in project directory or `Test Env/` folder

#### Option B: Convert from PyTorch
```bash
pip install torch depth-anything-v3

# Run model exporter (see model_exporter.py)
python model_exporter.py

# Convert to OpenVINO IR format using model optimizer
mo --input_model depth_anything_v2.onnx --output_dir .
```

## 🎮 Usage

### Basic Usage
```bash
python pathfinding_algorithm.py
```

This will:
1. Open your default webcam
2. Launch Rerun viewer in separate window
3. Display live 3D visualization with:
   - **Green points**: Navigable floor
   - **Red points**: Detected obstacles
   - **Blue line**: Computed safe path
   - **RGB overlay**: Path projected onto camera view

### Configuration

Edit constants in script header:

```python
# Depth range
DEPTH_MIN_M = 0.3       # Minimum depth (meters)
DEPTH_MAX_M = 4.0       # Maximum depth (meters)

# Floor detection
Y_BINS = 200            # Histogram resolution
FLOOR_SLAB_M = 0.08     # Floor thickness tolerance

# Obstacle detection
OBSTACLE_SLAB_M = 0.10  # Obstacle layer thickness
OBSTACLE_SLABS = 2      # Number of layers to check
MIN_OBSTACLE_POINTS = 3 # Minimum points to confirm obstacle

# Pathfinding
GRID_RESOLUTION = 0.1   # Occupancy grid cell size (meters)
GOAL_DISTANCE = 3.0     # Target distance ahead (meters)
```

### Advanced: Floor-Only Visualization
```bash
python Depth_estimation_floor_plane.py
```
Simplified version focusing on floor detection without pathfinding.

## 📊 Performance

Tested on **Intel Core i7-1165G7** (Iris Xe iGPU):

| Component | Latency | FPS |
|-----------|---------|-----|
| Camera Capture | ~33ms | 30 |
| GPU Inference | ~45ms | 22 |
| CPU Processing | ~15ms | 66 |
| Total Pipeline | ~93ms | **~10-15** |

**Bottleneck**: GPU inference (can be improved with model quantization)

### Optimization Tips
1. **Use Depth Anything V3 Small**: 2-3× faster than V2 ViT-B
2. **Quantize model to INT8**: Use OpenVINO's Post-Training Optimization Tool
3. **Reduce input resolution**: 256×256 instead of 518×518 (trades accuracy for speed)
4. **Increase queue depth**: Allows better pipelining but adds latency

## 🗂️ File Structure

```
Velo/
├── pathfinding_algorithm.py          # Main pipeline with A* pathfinding
├── Depth_estimation_floor_plane.py   # Simplified floor detection demo
├── model_exporter.py                  # PyTorch → OpenVINO conversion
├── depth_anything_v2_vitb.xml         # OpenVINO IR model (weights)
├── depth_anything_v2_vitb.bin         # OpenVINO IR model (graph)
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## 🔬 Algorithm Deep Dive

### 1. Depth to 3D Point Cloud

Pinhole camera model with ray-angle correction:

```python
# Pre-compute ray-scale factor (accounts for non-central rays)
_RAY_SCALE = sqrt((u - cx)²/fx² + (v - cy)²/fy² + 1)

# Back-project with correction
z_corrected = depth_map / _RAY_SCALE
x = (u - cx) / fx * z_corrected
y = (v - cy) / fy * z_corrected
```

### 2. Floor Detection

Y-axis histogram voting:
1. Bin all Y-coordinates into 200 buckets
2. Find peak (highest density) = floor level
3. Accept points within ±8cm of peak

**Why it works**: Floor is typically the largest planar surface, dominating the Y-histogram.

### 3. Obstacle Detection

Spatial XZ-binning with hash-based deduplication:

```python
# Hash XZ coordinates into 64-bit keys (collision-free)
key = int(x / 0.05) * 1000003 + int(z / 0.05)

# Count points per cell (vectorized)
unique_keys, counts = np.unique(keys, return_counts=True)
obstacle_cells = unique_keys[counts >= 3]
```

**Advantages**:
- Pure NumPy (no Python loops)
- Constant-time cell lookup
- Memory-efficient (stores only occupied cells)

### 4. A* Path Planning

Classic A* with grid snapping for robustness:

```python
f(n) = g(n) + h(n)
g(n) = cost from start to n
h(n) = Euclidean distance from n to goal
```

**Enhancements**:
- Diagonal movement (√2 cost)
- 2-iteration obstacle dilation for safety margin
- Automatic goal snapping to nearest free cell

### 5. Path Smoothing

Cubic B-spline fitting with controlled relaxation:

```python
# splprep parameters
s = len(waypoints) * 0.6    # Smoothing factor (higher = rounder)
k = 3                        # Cubic spline
u_new = linspace(0, 1, 120) # 120 interpolated points
```

**Benefit**: Converts jagged grid-based path into smooth curve suitable for motion planning.

## 🎯 Use Cases

1. **Assistive Navigation**: Guide visually impaired users through indoor environments
2. **Mobile Robotics**: Autonomous navigation for wheeled robots
3. **AR/VR**: Real-time spatial understanding for mixed reality apps
4. **Drone Landing**: Identify safe landing zones from aerial depth sensors
5. **Warehouse Automation**: Avoid dynamic obstacles in unstructured environments

## 🐛 Known Issues & Limitations

- **Staircase Challenge**: Floor detection assumes single horizontal plane (won't handle stairs)
- **Transparent Surfaces**: Depth models struggle with glass/mirrors
- **Thin Obstacles**: 5cm binning may miss thin poles/wires
- **Lighting Dependency**: Monocular depth estimation degrades in low light
- **Fixed Goal**: Currently hardcoded to 3m ahead (future: dynamic goal selection)

## 🔮 Future Roadmap

- [ ] SLAM integration for persistent occupancy mapping
- [ ] Multi-floor support (staircase detection)
- [ ] Dynamic obstacle tracking (Kalman filtering)
- [ ] ROS2 integration for robotic platforms
- [ ] Model quantization (INT8) for 2× speedup
- [ ] Semantic segmentation for walkable surface classification
- [ ] IMU fusion for improved odometry
- [ ] Web interface for remote monitoring

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Depth Anything V2/V3**: LiheYoung et al. for state-of-the-art monocular depth estimation
- **OpenVINO**: Intel for GPU-accelerated inference toolkit
- **Rerun**: Rerun.io for outstanding 3D visualization framework
- **SciPy**: For B-spline interpolation utilities

## 📧 Contact

**Maintainer**: [@ciada-3301](https://github.com/ciada-3301)

For questions, issues, or collaboration inquiries, please open an issue or reach out via GitHub.

---

**⭐ If you find this project useful, please consider starring the repository!**
