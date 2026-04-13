# Installation Guide

This guide provides detailed installation instructions for Velo across different platforms and use cases.

## Table of Contents
- [System Requirements](#system-requirements)
- [Platform-Specific Installation](#platform-specific-installation)
  - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
  - [Windows](#windows)
  - [macOS](#macos)
- [Model Setup](#model-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **GPU**: Intel integrated GPU (Gen9+) or any discrete GPU
- **Camera**: USB webcam or built-in camera (640×480 minimum)

### Recommended Requirements
- **RAM**: 8GB+
- **GPU**: Intel Iris Xe or discrete GPU
- **Camera**: 1080p webcam
- **CPU**: 4+ cores for optimal multi-threading

## Platform-Specific Installation

### Linux (Ubuntu/Debian)

#### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. Install System Dependencies
```bash
# Install Python and development tools
sudo apt install python3.10 python3-pip python3-venv -y

# Install OpenCV dependencies
sudo apt install libopencv-dev python3-opencv -y

# Install camera support
sudo apt install v4l-utils -y
```

#### 3. Clone Repository
```bash
git clone https://github.com/ciada-3301/Velo.git
cd Velo
```

#### 4. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 5. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 6. Verify Camera Access
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera (should show video)
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Windows

#### 1. Install Python
Download and install [Python 3.10+](https://www.python.org/downloads/) (check "Add Python to PATH")

#### 2. Install Git
Download and install [Git for Windows](https://git-scm.com/download/win)

#### 3. Clone Repository
```powershell
git clone https://github.com/ciada-3301/Velo.git
cd Velo
```

#### 4. Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

#### 5. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 6. Install Visual C++ Redistributables (if needed)
Download from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) if you encounter DLL errors.

### macOS

#### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Python
```bash
brew install python@3.10
```

#### 3. Clone Repository
```bash
git clone https://github.com/ciada-3301/Velo.git
cd Velo
```

#### 4. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 5. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Setup

### Option A: Download Pre-converted Model (Recommended)

**Coming Soon**: Pre-converted OpenVINO IR models will be available in releases.

For now, proceed to Option B to convert the model yourself.

### Option B: Convert Model from Source

#### 1. Install PyTorch and Depth Anything V3
```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install PyTorch (CPU version - GPU not needed for export)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Depth Anything V3
pip install git+https://github.com/DepthAnything/Depth-Anything-V3.git
```

#### 2. Download PyTorch Weights
```python
# Run this Python script to download model
python -c "
from depth_anything_v3.dpt import DepthAnythingV3
model = DepthAnythingV3.from_pretrained('depth-anything/DA3-SMALL')
print('Model downloaded successfully!')
"
```

#### 3. Export to ONNX (using model_exporter.py)
Create an export script:

```python
import torch
from depth_anything_v3.dpt import DepthAnythingV3

device = 'cpu'
model = DepthAnythingV3.from_pretrained("depth-anything/DA3-SMALL").to(device).eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 518, 518)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "depth_anything_v3_small.onnx",
    opset_version=14,
    input_names=['input'],
    output_names=['depth'],
    dynamic_axes={'input': {0: 'batch'}, 'depth': {0: 'batch'}}
)
print("ONNX export complete!")
```

Save as `export_model.py` and run:
```bash
python export_model.py
```

#### 4. Convert ONNX to OpenVINO IR
```bash
# Install OpenVINO Model Optimizer
pip install openvino-dev

# Convert to OpenVINO IR format
mo --input_model depth_anything_v3_small.onnx \
   --output_dir . \
   --model_name depth_anything_v2_vitb \
   --compress_to_fp16
```

This creates `depth_anything_v2_vitb.xml` and `depth_anything_v2_vitb.bin`.

#### 5. Update Path in Code
Edit `pathfinding_algorithm.py` line 18:
```python
compiled_model = core.compile_model("depth_anything_v2_vitb.xml", "GPU")
```

## Verification

### Test Installation
```bash
python pathfinding_algorithm.py
```

You should see:
1. Camera feed window opens
2. Rerun viewer launches
3. Console output: `Live feed started...`
4. Point cloud visualization appears

### Common Issues

#### Camera Not Found
```bash
# Linux: Check permissions
sudo usermod -a -G video $USER
# Logout and login again

# Check camera index
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### OpenVINO GPU Error
```bash
# Check available devices
python -c "
import openvino as ov
core = ov.Core()
print('Available devices:', core.available_devices)
"

# If GPU not listed, install GPU drivers:
# Intel: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
```

#### Import Errors
```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### Rerun Window Doesn't Open
```bash
# Install/update Rerun viewer
pip install --upgrade rerun-sdk

# Test Rerun separately
python -c "
import rerun as rr
rr.init('test', spawn=True)
rr.log('test', rr.TextLog('Hello!'))
import time; time.sleep(2)
"
```

## Performance Optimization

### Intel GPU Setup (Linux)
```bash
# Install Intel Compute Runtime
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero -y

# Verify GPU is detected
clinfo | grep "Device Name"
```

### For Laptops: Prevent Thermal Throttling
```bash
# Install TLP (power management)
sudo apt install tlp tlp-rdw -y
sudo tlp start

# Monitor temperatures
watch -n 1 sensors
```

## Next Steps

After successful installation:
1. Read the [README.md](README.md) for usage instructions
2. Experiment with configuration parameters
3. Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
4. Join discussions in GitHub Issues

## Getting Help

If you encounter issues:
1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/ciada-3301/Velo/issues)
3. Create a new issue with:
   - OS and Python version
   - Full error message
   - Steps to reproduce

Happy navigating! 🚀
