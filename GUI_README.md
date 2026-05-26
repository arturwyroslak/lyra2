# Lyra 2.0 GUI - Interactive Scene Generation

## Overview

The Lyra 2.0 GUI provides a professional, interactive interface for generating explorable 3D worlds from single images. The interface is built with Gradio and follows a professional 3D software layout similar to Blender or Unreal Engine.

## Features

### Three-Panel Layout

**Left Panel - Camera Path & Video Generator**
- Camera position and rotation controls (X, Y, Z, Pitch, Yaw, Roll)
- Camera path recording, clearing, and smoothing
- Camera view preview window
- Video generation controls:
  - Camera trajectory selection (30+ trajectory types)
  - Direction control (left, right, up, down)
  - Trajectory strength slider
  - Generate length (frames)
  - Prompt type (Text/Image)
  - Text prompt input
  - DMD fast inference option
- Progress bar and status display
- Generate Video and Reconstruct 3D Scene buttons

**Center Panel - Main Viewport**
- Tabbed interface for different views:
  - Input Image - Upload and view source image
  - Depth Map - Visualize estimated depth
  - Generated Video - View AI-generated video
  - 3D Reconstruction - View reconstructed 3D scene
- Viewport controls (Reset, Fit, Toggle Grid)

**Right Panel - Scene Hierarchy**
- Scene structure tree view
- Layer visibility controls (RGB, Depth, Normals, Trajectory)
- Layer opacity slider
- Scene statistics (point count, bounds, GPU memory)
- Export options (PLY, OBJ, GLTF formats)

## Installation

### Quick Install with Script

Run the provided installation script:

```bash
cd Lyra-2
bash install_with_gui.sh
```

This script will:
1. Create a conda environment with Python 3.10
2. Install CUDA 12.8 toolkit
3. Install PyTorch 2.7.1
4. Install all required dependencies
5. Build CUDA extensions
6. Install GUI dependencies (Gradio, MoviePy)
7. Configure environment variables
8. Verify installation

### Manual Installation

Follow the standard [INSTALL.md](INSTALL.md) instructions, then add:

```bash
pip install gradio>=4.0.0
pip install moviepy>=1.0.0
```

## Usage

### Starting the GUI

1. Set environment variables:
```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

2. Launch the GUI:
```bash
PYTHONPATH=. python3 -m lyra_2._src.gui.lyra_gui
```

3. Open your browser to: `http://localhost:7860`

**For Google Colab:**
```python
# Set environment variables
%env CUDA_HOME=/usr/local/cuda
%env LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Run GUI with public URL
PYTHONPATH=. python3 -m lyra_2._src.gui.lyra_gui --share
```

### Workflow

1. **Load Models**: Click "Load Models" in the Model Settings accordion. This loads the Lyra2 model, DA3 depth model, and MoGe model.

2. **Upload Image**: In the "Input Image" tab, upload a single image you want to explore.

3. **Process Image**: Click "Process Image" to estimate depth and initialize the scene.

4. **Configure Generation**:
   - Enter a text prompt describing the scene
   - Select camera trajectory (e.g., "horizontal_zoom")
   - Choose direction (right = forward, left = backward)
   - Adjust trajectory strength
   - Set number of frames (must be 1 + 80k, e.g., 81, 161, 241)
   - Optionally enable DMD fast inference for ~15x speedup

5. **Generate Video**: Click "Generate Video" to create an exploration video along the camera path.

6. **Reconstruct 3D**: Click "Reconstruct 3D Scene" to lift the video to a 3D Gaussian Splatting scene.

7. **Export**: Use export buttons to save the scene in various formats.

## Camera Trajectories

The GUI supports 30+ camera trajectory types:

- **Basic**: horizontal, vertical, horizontal_simple, vertical_simple
- **Zoom**: horizontal_zoom, horizontal_zoom_noise, horizontal_zoom_bend, dolly_zoom
- **Spiral**: spiral, spiral_center, spiral_outwards, horizontal_spiral
- **Orbit**: orbit_horizontal, orbit_vertical
- **Rotate**: rotate_zoom_in, rotate_zoom_out, rotate_spot, rotate_spot_noise
- **Complex**: horizontal_zoom_noise_bend, horizontal_lift, horizontal_lift_noise
- And more...

## DMD Fast Inference

Enable "Use DMD Fast Inference" to:
- Reduce sampling to 4 steps (from 50)
- Disable CFG (classifier-free guidance)
- Achieve ~15x speedup
- Trade-off: slightly weaker prompt following, possible repetitive patterns

## Requirements

- Ubuntu 22.04 (or similar Linux distribution)
- CUDA 12.4+ (tested with CUDA 12.8)
- NVIDIA GPU with sufficient VRAM (H100 80GB recommended)
- Python 3.10
- pip package manager

## Troubleshooting

### Model Loading Issues
- Ensure checkpoints are downloaded from HuggingFace:
  ```bash
  huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
  ```
- Verify `checkpoints/model` directory exists

### CUDA Errors
- Ensure CUDA_HOME is set correctly
- Check LD_LIBRARY_PATH includes CUDA libraries
- Verify GPU drivers are up to date

### GUI Not Starting
- Check that Gradio is installed: `pip show gradio`
- Verify port 7860 is not in use
- Check firewall settings

### Memory Issues
- Enable model offloading in Model Settings
- Reduce number of frames
- Use DMD fast inference
- Close other GPU-intensive applications

## Advanced Usage

### Custom Camera Paths
For custom camera trajectories, prepare a `trajectory.npz` file with:
- `w2c`: N×4×4 world-to-camera matrices
- `intrinsics`: N×3×3 camera intrinsics
- `image_height`, `image_width`: resolution

Use the custom trajectory inference script instead of the GUI for this.

### Batch Processing
The GUI is designed for interactive single-image processing. For batch processing, use the command-line scripts:
- `lyra2_zoomgs_inference.py` - Zoom-in/zoom-out generation
- `lyra2_custom_traj_inference.py` - Custom trajectory generation
- `vipe_da3_gs_recon.py` - 3D reconstruction

## Architecture

The GUI is built with:
- **Gradio 4.0+**: Web interface framework
- **PyTorch**: Deep learning backend
- **OpenCV**: Image processing
- **MoviePy**: Video export
- **Lyra2 Core**: Video generation model
- **DA3**: Depth estimation
- **VIPE**: Pose estimation
- **3D Gaussian Splatting**: 3D reconstruction

## License

Same as Lyra 2.0:
- Source code: Apache 2.0 License
- Models: NVIDIA Internal Scientific Research and Development Model License

## Support

For issues specific to the GUI, please check:
1. Installation is complete (run verification steps)
2. Checkpoints are downloaded
3. Environment variables are set correctly
4. GPU has sufficient memory

For general Lyra 2.0 issues, refer to the main README and INSTALL.md.
