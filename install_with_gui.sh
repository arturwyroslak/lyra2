#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installation script for Lyra 2.0 with GUI support (No Conda)
# Based on INSTALL.md with additional GUI dependencies

set -e  # Exit on error

echo "=========================================="
echo "Lyra 2.0 Installation with GUI Support"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -ne 10 ]; then
    echo "Warning: Python 3.10 is recommended. Current version: $PYTHON_VERSION"
    echo "Installation may fail with other Python versions."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA
echo ""
echo "Checking CUDA installation..."
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Found CUDA at $CUDA_HOME"
    else
        echo "Error: CUDA_HOME not set and /usr/local/cuda not found"
        echo "Please install CUDA 12.4+ and set CUDA_HOME"
        exit 1
    fi
fi

CUDA_VERSION=$($CUDA_HOME/bin/nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
echo "CUDA version: $CUDA_VERSION"


# Step 1: Install system dependencies
echo ""
echo "Step 1: Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv cmake ninja-build libgl1-mesa-glx ffmpeg pkg-config libeigen3-dev zlib1g-dev
elif command -v yum &> /dev/null; then
    sudo yum install -y python3-pip cmake ninja-build mesa-libGL ffmpeg pkgconfig eigen3-devel zlib-devel
else
    echo "Warning: Could not detect package manager. Please install: cmake, ninja-build, libgl, ffmpeg, pkg-config, eigen3, zlib"
fi

# Step 2: Upgrade pip and install build tools
echo ""
echo "Step 2: Upgrading pip and installing build tools..."
python3 -m pip install --upgrade pip setuptools wheel

# Step 3: Install PyTorch
echo ""
echo "Step 3: Installing PyTorch..."
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# Step 4: Set build environment variables
echo ""
echo "Step 4: Setting build environment variables..."
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
export CPATH="$CUDA_HOME/include:$CPATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=$CUDA_HOME

# Step 5: Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies..."
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"

# Symlink cuda_runtime as cudart for transformer_engine compatibility
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart" 2>/dev/null || true

# Step 6: Install Flash Attention
echo ""
echo "Step 6: Installing Flash Attention..."
MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# Step 7: Build vendored CUDA extensions
echo ""
echo "Step 7: Building vendored CUDA extensions..."
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# Step 8: Install GUI dependencies
echo ""
echo "Step 8: Installing GUI dependencies (Gradio)..."
pip install gradio>=4.0.0
pip install moviepy>=1.0.0
pip install pillow>=10.0.0
pip install requests>=2.31.0

# Add LD_LIBRARY_PATH to shell profile
echo ""
echo "Step 9: Adding LD_LIBRARY_PATH to shell profile..."
SHELL_PROFILE="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
fi

if ! grep -q "LYRA2_LD_LIBRARY_PATH" "$SHELL_PROFILE" 2>/dev/null; then
    echo "" >> "$SHELL_PROFILE"
    echo "# Lyra 2.0 LD_LIBRARY_PATH" >> "$SHELL_PROFILE"
    echo "export LYRA2_LD_LIBRARY_PATH=true" >> "$SHELL_PROFILE"
    echo "export CUDA_HOME=/usr/local/cuda" >> "$SHELL_PROFILE"
    echo "export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\"" >> "$SHELL_PROFILE"
    echo "Added LD_LIBRARY_PATH to $SHELL_PROFILE"
else
    echo "LD_LIBRARY_PATH already configured in $SHELL_PROFILE"
fi

# Step 10: Download checkpoints
echo ""
echo "Step 10: Downloading checkpoints..."
echo "Please ensure you have downloaded the checkpoints from HuggingFace:"
echo "  huggingface-cli download nvidia/Lyra-2.0 --include \"checkpoints/*\" --local-dir ."
echo ""
read -p "Have you downloaded the checkpoints? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please download checkpoints manually from:"
    echo "  https://huggingface.co/nvidia/Lyra-2.0"
    echo ""
    echo "Run: huggingface-cli download nvidia/Lyra-2.0 --include \"checkpoints/*\" --local-dir ."
fi

# Step 11: Verify installation
echo ""
echo "Step 11: Verifying installation..."
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PYTHONPATH=. python3 -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

PYTHONPATH=. python3 -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python3 -m lyra_2._src.inference.vipe_da3_gs_recon --help

# Verify GUI dependencies
echo ""
echo "Verifying GUI dependencies..."
python3 -c "
import gradio
print('gradio:', gradio.__version__)
print('GUI dependencies OK')
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To use Lyra 2.0 GUI:"
echo "  1. Set environment variables:"
echo "     export CUDA_HOME=/usr/local/cuda"
echo "     export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\""
echo "  2. Run the GUI:"
echo "     PYTHONPATH=. python3 -m lyra_2._src.gui.lyra_gui"
echo ""
echo "The GUI will be available at http://localhost:7860"
echo ""
