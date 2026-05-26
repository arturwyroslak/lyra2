# Lyra 2.0 Street View GUI - Google Street View to 3D

## Overview

The Lyra 2.0 Street View GUI provides a specialized interface for converting Google Street View panoramas into explorable 3D scenes. This tool allows you to search for real-world locations, download 360° panoramic views, and use Lyra 2.0's generative capabilities to expand and reconstruct these locations as interactive 3D environments.

## Features

### Street View Integration

**Location Search & Download:**
- Search for locations by address or coordinates
- Google Street View API integration
- Download multiple directional views for 360° coverage
- Configurable field of view (FOV) and number of directions
- Automatic panorama metadata retrieval

**View Selection:**
- Gallery view of all downloaded directions
- Individual view selection for processing
- Depth estimation for each view
- Camera trajectory planning from selected viewpoint

### Three-Panel Layout

**Left Panel - Street View Controls:**
- Google Street View API key configuration
- Location search with address or coordinates
- Download settings (directions, FOV)
- View gallery with 4-column grid
- View selection slider
- Download status and metadata display

**Center Panel - Viewport & Processing:**
- Tabbed interface:
  - Selected View - Display chosen Street View image
  - Depth Map - Visualize estimated depth
  - Generated Video - View AI-generated expansion
  - 3D Reconstruction - View reconstructed 3D scene
- Video generation controls
- Progress tracking

**Right Panel - Scene Information:**
- Location details (coordinates, panorama ID)
- Generation statistics (trajectory, time, quality)
- Export options (PLY, OBJ, GLTF)
- Tips and usage guidance

## Installation

### Google Street View API Key

To use the Street View functionality, you need a Google Street View API key:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Street View Static API
4. Create an API key with appropriate restrictions
5. Copy the API key for use in the GUI

**Note:** The GUI can run in demo mode without an API key, but functionality will be limited.

### Installation with Script

Run the installation script (includes Street View dependencies):

```bash
cd Lyra-2
bash install_with_gui.sh
```

### Manual Installation

Follow the standard [INSTALL.md](INSTALL.md) instructions, then add:

```bash
pip install gradio>=4.0.0
pip install moviepy>=1.0.0
pip install requests>=2.31.0
```

## Usage

### Starting the Street View GUI

1. Activate the conda environment:
```bash
conda activate lyra2
```

2. Set environment variables:
```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

3. Launch the Street View GUI:
```bash
PYTHONPATH=. python -m lyra_2._src.gui.streetview_gui
```

4. Open your browser to: `http://localhost:7861`

**Note:** The Street View GUI runs on port 7861 to avoid conflicts with the main GUI (port 7860).

### Workflow

1. **Configure API Key:**
   - Enter your Google Street View API key
   - Click "Set API Key"
   - Verify status shows "API key set successfully"

2. **Search Location:**
   - Enter a location (e.g., "Eiffel Tower, Paris" or "48.8584,2.2945")
   - Click "Search Location"
   - Review search results and coordinates

3. **Download Views:**
   - Adjust number of directions (4-16 for 360° coverage)
   - Set field of view (60-120 degrees)
   - Click "Download Views"
   - Wait for all directional images to download
   - View gallery will populate with downloaded images

4. **Select View:**
   - Use the slider to select a view from the gallery
   - Click "Select View"
   - The selected view appears in the viewport

5. **Process View:**
   - Click "Process View" to estimate depth
   - Review the depth map visualization
   - Depth estimation is required before generation

6. **Configure Generation:**
   - Enter a prompt describing scene expansion
   - Select camera trajectory (e.g., "horizontal_zoom")
   - Choose direction (right = forward, left = backward)
   - Adjust trajectory strength
   - Set number of frames (must be 1 + 80k, e.g., 81, 161, 241)
   - Optionally enable DMD fast inference

7. **Generate Video:**
   - Click "Generate Video"
   - Monitor progress bar
   - View generated video in the Video tab

8. **Reconstruct 3D:**
   - Click "Reconstruct 3D Scene"
   - Wait for VIPE pose estimation and DA3 reconstruction
   - View 3D reconstruction in the 3D Reconstruction tab

9. **Export:**
   - Use export buttons to save scene or video
   - Choose export format (PLY, OBJ, GLTF)

## Location Search Formats

You can search for locations in multiple formats:

**Address:**
- "Eiffel Tower, Paris, France"
- "Times Square, New York, NY"
- "1600 Amphitheatre Parkway, Mountain View, CA"

**Coordinates:**
- "48.8584,2.2945" (Eiffel Tower)
- "40.7580,-73.9855" (Times Square)
- "37.422,-122.084" (Google HQ)

**Place IDs:**
- Google Place IDs (advanced usage)

## Download Settings

**Number of Directions:**
- 4 directions: 90° spacing (basic coverage)
- 8 directions: 45° spacing (recommended)
- 12 directions: 30° spacing (high coverage)
- 16 directions: 22.5° spacing (maximum coverage)

More directions provide better 360° coverage but take longer to download.

**Field of View (FOV):**
- 60°: Narrow, more detail per view
- 90°: Standard (recommended)
- 120°: Wide, more coverage per view

Lower FOV provides more detail but requires more views for full coverage.

## Camera Trajectories

The Street View GUI supports the same 30+ camera trajectory types as the main GUI:

- **Basic:** horizontal, vertical, horizontal_simple, vertical_simple
- **Zoom:** horizontal_zoom, horizontal_zoom_noise, dolly_zoom
- **Spiral:** spiral, spiral_center, spiral_outwards
- **Orbit:** orbit_horizontal, orbit_vertical
- **Rotate:** rotate_zoom_in, rotate_zoom_out, rotate_spot
- And more...

For Street View scenes, recommended trajectories:
- `horizontal_zoom` - Walk forward from viewpoint
- `spiral` - Spiral exploration around viewpoint
- `orbit_horizontal` - Orbit around the scene
- `horizontal_simple` - Simple sideways movement

## Use Cases

### Virtual Tourism
- Explore famous landmarks in 3D
- Walk through streets and neighborhoods
- Create virtual tours from real locations

### Urban Planning
- Visualize city expansions
- Plan new developments in context
- Analyze street-level perspectives

### Real Estate
- Generate 3D models from Street View
- Explore neighborhoods remotely
- Create property visualizations

### Game Development
- Use real-world locations as game environments
- Generate realistic urban scenes
- Create location-based AR/VR content

### Research
- Study urban environments
- Analyze street-level data
- Create 3D datasets from Street View

## API Key Management

### Getting an API Key

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Navigate to APIs & Services > Library
4. Search for "Street View Static API"
5. Click "Enable"
6. Go to APIs & Services > Credentials
7. Click "Create Credentials" > "API Key"
8. (Optional) Set application restrictions and API restrictions
9. Copy the API key

### API Key Security

- Never commit API keys to version control
- Use environment variables for production deployments
- Set application restrictions in Google Cloud Console
- Monitor usage in Google Cloud Console
- Set daily quotas to prevent abuse

### Quotas and Limits

Google Street View API has free tier limits:
- Free tier: 25,000 requests per day
- Beyond free tier: $2 per 1,000 requests

Monitor your usage in Google Cloud Console to avoid unexpected charges.

## Troubleshooting

### API Key Issues
- **"API key not valid"**: Verify the key is correct and enabled
- **"Quota exceeded"**: Check Google Cloud Console for usage
- **"Forbidden"**: Verify API restrictions are properly configured

### Download Issues
- **"No Street View found"**: Location may not have Street View coverage
- **"Failed to download"**: Check internet connection and API key
- **"Rate limited"**: Wait a few minutes and try again

### Generation Issues
- **"Please process a view first"**: Click "Process View" before generating
- **"Please enter a prompt"**: Enter a text prompt for generation
- **CUDA out of memory**: Reduce number of frames or enable model offloading

### Port Conflicts
- If port 7861 is in use, modify the port in `streetview_gui.py`:
  ```python
  interface.launch(server_port=7862)  # Change to available port
  ```

## Advanced Usage

### Batch Processing

To process multiple locations, create a script:

```python
from lyra_2._src.gui.streetview_gui import StreetViewGUI

locations = [
    "Eiffel Tower, Paris",
    "Times Square, New York",
    "Taj Mahal, Agra"
]

gui = StreetViewGUI()
gui.set_api_key("your-api-key")
gui.load_models()

for location in locations:
    coords = gui.search_location(location)
    views = gui.download_views(coords, num_directions=8)
    # Process and generate...
```

### Custom Panorama Sources

You can modify the `StreetViewDownloader` class to use other panorama sources:
- Mapillary
- OpenStreetView
- Custom panorama datasets

### Integration with Main GUI

The Street View GUI uses the same Lyra 2.0 backend as the main GUI. You can:
- Use the same models and checkpoints
- Share generated outputs between GUIs
- Combine Street View with custom images

## Comparison: Street View GUI vs Main GUI

| Feature | Street View GUI | Main GUI |
|---------|----------------|----------|
| Input Source | Google Street View | Custom images |
| Location Search | Yes | No |
| 360° Views | Yes | No |
| API Required | Google Street View API | None |
| Best For | Real-world locations | Custom scenes |
| Port | 7861 | 7860 |

## Requirements

- Ubuntu 22.04 (or similar Linux distribution)
- CUDA 12.4+ (tested with CUDA 12.8)
- NVIDIA GPU with sufficient VRAM (H100 80GB recommended)
- Python 3.10
- Conda environment
- Google Street View API key (optional for demo mode)
- Internet connection for Street View downloads

## License

Same as Lyra 2.0:
- Source code: Apache 2.0 License
- Models: NVIDIA Internal Scientific Research and Development Model License
- Google Street View data: Subject to Google Maps Terms of Service

## Terms of Service

When using Google Street View:
- Comply with Google Maps Terms of Service
- Respect usage quotas and limits
- Do not use for prohibited purposes
- Attribute Google when displaying Street View data
- Check local laws regarding photography and privacy

## Support

For Street View-specific issues:
1. Verify API key is valid and enabled
2. Check Google Cloud Console for quota limits
3. Ensure location has Street View coverage
4. Verify internet connection

For general Lyra 2.0 issues, refer to the main README and INSTALL.md.

## Future Enhancements

Planned features for future versions:
- Automatic panorama stitching
- Multi-location batch processing
- Integration with other map providers
- Custom panorama upload support
- Advanced trajectory planning
- Real-time preview during generation
