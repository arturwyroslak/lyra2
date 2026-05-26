#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installation script for Lyra 2.0 with GUI support
# Based on INSTALL.md with additional GUI dependencies

set -e  # Exit on error

echo "=========================================="
echo "Lyra 2.0 Installation with GUI Support"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Step 0: Clone repository if not already in it
if [ ! -f "INSTALL.md" ]; then
    echo "Step 0: Cloning repository..."
    git clone --recursive git@github.com:nv-tlabs/lyra.git Lyra-2
    cd Lyra-2
else
    echo "Step 0: Already in repository directory"
fi

# Step 1: Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
conda activate lyra2
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y

# Step 2: Install CUDA toolkit inside the conda environment
echo ""
echo "Step 2: Installing CUDA toolkit..."
conda install cuda -c nvidia/label/cuda-12.8.0 -y
export CUDA_HOME=$CONDA_PREFIX

# Step 3: Install PyTorch
echo ""
echo "Step 3: Installing PyTorch..."
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# Step 4: Set build environment variables
echo ""
echo "Step 4: Setting build environment variables..."
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# Step 5: Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies..."
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"

# Symlink cuda_runtime as cudart for transformer_engine compatibility
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

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
    echo "SITE=\$CONDA_PREFIX/lib/python3.10/site-packages" >> "$SHELL_PROFILE"
    echo "export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\$SITE/torch/lib:\$SITE/nvidia/cuda_runtime/lib:\$SITE/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" >> "$SHELL_PROFILE"
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
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help

# Verify GUI dependencies
echo ""
echo "Verifying GUI dependencies..."
python -c "
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
echo "  1. Activate the conda environment: conda activate lyra2"
echo "  2. Set environment variables:"
echo "     SITE=\$CONDA_PREFIX/lib/python3.10/site-packages"
echo "     export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\$SITE/torch/lib:\$SITE/nvidia/cuda_runtime/lib:\$SITE/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
echo "  3. Run the GUI:"
echo "     PYTHONPATH=. python -m lyra_2._src.gui.lyra_gui"
echo ""
echo "The GUI will be available at http://localhost:7860"
echo ""
