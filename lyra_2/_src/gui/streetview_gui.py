# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lyra 2.0 Street View GUI
Interface for converting Google Street View panoramas to explorable 3D scenes.
"""

import os
import sys
import gc
import json
import tempfile
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import threading
import queue
import math

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
import requests

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lyra_2._src.inference.camera_traj_utils import CAMERA_TRAJECTORY_CHOICES
from lyra_2._src.utils.model_loader import load_model_from_checkpoint
from lyra_2._src.inference.depth_utils import load_da3_model, load_moge_model
from lyra_2._ext.imaginaire.utils import log, misc


class StreetViewDownloader:
    """Download and process Google Street View panoramas."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    
    def get_panorama_metadata(self, location: str) -> Optional[Dict]:
        """Get metadata for a Street View location."""
        params = {
            "location": location,
            "key": self.api_key
        }
        
        try:
            response = requests.get(self.metadata_url, params=params)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return None
    
    def download_panorama(
        self,
        location: str,
        size: Tuple[int, int] = (640, 640),
        fov: int = 90,
        heading: float = 0,
        pitch: float = 0
    ) -> Optional[np.ndarray]:
        """Download a Street View image."""
        params = {
            "location": location,
            "size": f"{size[0]}x{size[1]}",
            "fov": fov,
            "heading": heading,
            "pitch": pitch,
            "key": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"Error downloading panorama: {e}")
            return None
    
    def download_360_panorama(
        self,
        location: str,
        size: Tuple[int, int] = (2048, 1024),
        num_directions: int = 8
    ) -> List[np.ndarray]:
        """Download multiple views to create a 360 panorama."""
        images = []
        headings = np.linspace(0, 360, num_directions, endpoint=False)
        
        for heading in headings:
            img = self.download_panorama(location, size=size, heading=heading)
            if img is not None:
                images.append(img)
        
        return images


class StreetViewGUI:
    """Main GUI class for Street View to 3D conversion."""
    
    def __init__(self):
        self.model = None
        self.da3_model = None
        self.moge_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = "checkpoints/model"
        self.output_dir = "outputs/streetview"
        self.downloader = None
        self.current_panorama = None
        self.current_views = []
        self.selected_view = None
        self.current_depth = None
        self.current_intrinsics = None
        self.generated_video_path = None
        self.reconstructed_scene_path = None
        self.api_key = ""
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def set_api_key(self, api_key: str) -> str:
        """Set Google Street View API key."""
        self.api_key = api_key
        self.downloader = StreetViewDownloader(api_key=api_key if api_key else None)
        return "API key set successfully" if api_key else "API key cleared (using demo mode)"
    
    def load_models(self, use_dmd: bool = False, offload: bool = False) -> str:
        """Load Lyra2 model and depth models."""
        try:
            status = "Loading models...\n"
            
            # Load negative prompt embeddings
            negative_prompt_path = "checkpoints/text_encoder/negative_prompt.pt"
            if not os.path.exists(negative_prompt_path):
                return f"Error: Negative prompt file not found at {negative_prompt_path}"
            
            self.negative_prompt_data = torch.load(
                negative_prompt_path, map_location="cpu", weights_only=False
            )
            status += "✓ Negative prompt embeddings loaded\n"
            
            # Load Lyra2 model
            experiment_opts = [
                "model.config.use_mp_policy_fsdp=False",
                "model.config.keep_original_net_dtype=False",
            ]
            
            lora_paths = ["checkpoints/lora/realism_boost.safetensors",
                         "checkpoints/lora/detail_enhancer.safetensors"]
            lora_weights = [0.4, 0.4]
            
            if use_dmd:
                lora_paths.append("checkpoints/lora/dmd_distillation.safetensors")
                lora_weights.append(1.0)
                experiment_opts += ["model.config.net.postpone_checkpoint=True"]
                status += "✓ DMD distillation enabled\n"
            
            if offload:
                experiment_opts += ["model.config.net.postpone_checkpoint=True"]
            
            self.model, config = load_model_from_checkpoint(
                config_file="lyra_2/_src/configs/config.py",
                experiment_name="lyra2",
                checkpoint_path=self.checkpoint_dir,
                enable_fsdp=False,
                instantiate_ema=False,
                load_ema_to_reg=False,
                experiment_opts=experiment_opts,
            )
            
            # Load LoRA weights
            if lora_paths:
                lora_names = []
                for lora_path in lora_paths:
                    if os.path.exists(lora_path):
                        lora_name = self.model.load_lora_weights(lora_path)
                        lora_names.append(lora_name)
                if lora_names:
                    self.model.set_weights_and_activate_adapters(lora_names, lora_weights)
                    if hasattr(self.model, "net") and hasattr(self.model.net, "enable_selective_checkpoint"):
                        self.model.net.enable_selective_checkpoint(
                            self.model.net.sac_config, self.model.net.blocks
                        )
            
            desired_dtype = self.model.tensor_kwargs.get("dtype", None)
            desired_device = self.model.tensor_kwargs.get("device", None)
            if desired_dtype is not None:
                self.model.net = self.model.net.to(device=desired_device, dtype=desired_dtype)
            
            self.model.eval()
            status += f"✓ Lyra2 model loaded on {desired_device}\n"
            
            # Load DA3 model
            self.da3_model = load_da3_model(
                da3_model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                da3_model_path_custom="checkpoints/recon/model.pt",
                device=str(self.device),
            )
            self.da3_model.eval()
            status += f"✓ DA3 depth model loaded\n"
            
            # Load MoGe model
            self.moge_model = load_moge_model(self.device)
            self.moge_model.eval()
            status += f"✓ MoGe model loaded\n"
            
            status += "\n✓ All models loaded successfully!"
            return status
            
        except Exception as e:
            return f"Error loading models: {str(e)}"
    
    def search_location(self, location: str) -> Tuple[Optional[str], str]:
        """Search for a Street View location."""
        if not location:
            return None, "Please enter a location"
        
        if not self.downloader:
            return None, "Please set API key first"
        
        metadata = self.downloader.get_panorama_metadata(location)
        
        if metadata and metadata.get("status") == "OK":
            pano_id = metadata.get("pano_id", "")
            location_str = metadata.get("location", {}).get("latLng", {})
            lat = location_str.get("lat", 0)
            lng = location_str.get("lng", 0)
            
            return f"Found panorama: {pano_id}\nLocation: {lat}, {lng}", f"{lat},{lng}"
        else:
            return None, f"No Street View found for: {location}"
    
    def download_views(
        self,
        location_coords: str,
        num_directions: int,
        fov: int,
        progress=gr.Progress()
    ) -> Tuple[List[np.ndarray], str]:
        """Download multiple views from Street View."""
        if not location_coords:
            return [], "Please search for a location first"
        
        if not self.downloader:
            return [], "Please set API key first"
        
        try:
            progress(0, desc="Downloading Street View images...")
            
            self.current_views = []
            headings = np.linspace(0, 360, num_directions, endpoint=False)
            
            for i, heading in enumerate(headings):
                progress(i / num_directions, desc=f"Downloading view {i+1}/{num_directions}...")
                
                img = self.downloader.download_panorama(
                    location_coords,
                    size=(640, 640),
                    fov=fov,
                    heading=float(heading),
                    pitch=0
                )
                
                if img is not None:
                    self.current_views.append(img)
            
            progress(1.0, desc="Complete!")
            
            if self.current_views:
                return self.current_views, f"Downloaded {len(self.current_views)} views"
            else:
                return [], "Failed to download any views"
            
        except Exception as e:
            return [], f"Error downloading views: {str(e)}"
    
    def select_view(self, idx: int) -> Tuple[np.ndarray, str]:
        """Select a specific view for processing."""
        if 0 <= idx < len(self.current_views):
            self.selected_view = self.current_views[idx]
            return self.selected_view, f"Selected view {idx+1}"
        return np.zeros((100, 100, 3), dtype=np.uint8), "Invalid view index"
    
    def process_view(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process selected view and estimate depth."""
        if image is None:
            return None, "Please select a view first"
        
        try:
            self.current_image = image
            
            # Convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Estimate depth using DA3
            H, W = 480, 832
            img_resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            image_chw01 = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
            image_chw01 = image_chw01.permute(2, 0, 1).unsqueeze(0).contiguous()
            
            images = [img_resized.astype(np.uint8)]
            prediction = self.da3_model.inference(
                image=images,
                extrinsics=None,
                intrinsics=None,
                align_to_input_ext_scale=True,
                infer_gs=False,
                process_res=int(max(H, W)),
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="mini_npz",
            )
            
            depths_np = getattr(prediction, "depth", None)
            if depths_np is None:
                raise RuntimeError("DA3 prediction has no 'depth' field.")
            
            if isinstance(depths_np, torch.Tensor):
                depth_np = depths_np[0].detach().cpu().numpy()
            else:
                depth_np = np.asarray(depths_np)[0]
            
            depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if depth_t.shape[-2:] != (H, W):
                import torch.nn.functional as F
                depth_t = F.interpolate(depth_t, size=(H, W), mode="bilinear", align_corners=False)
            
            self.current_depth = depth_t[0, 0]
            self.current_depth = torch.nan_to_num(self.current_depth, nan=1e4).clamp(min=0, max=1e4)
            
            # Get intrinsics
            try:
                ixts_np = getattr(prediction, "intrinsics", None)
                if ixts_np is None:
                    raise AttributeError
                if isinstance(ixts_np, torch.Tensor):
                    K_np = ixts_np[0].detach().cpu().numpy()
                else:
                    K_np = np.asarray(ixts_np)[0]
                self.current_intrinsics = torch.from_numpy(K_np.astype(np.float32))
            except Exception:
                fx = fy = max(H, W) * 1.5
                cx, cy = W / 2.0, H / 2.0
                self.current_intrinsics = torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], 
                    dtype=torch.float32
                )
            
            # Visualize depth
            depth_vis = self.current_depth.cpu().numpy()
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
            
            return depth_vis, f"View processed successfully. Resolution: {H}x{W}"
            
        except Exception as e:
            return None, f"Error processing view: {str(e)}"
    
    def generate_video(
        self,
        prompt: str,
        trajectory: str,
        direction: str,
        strength: float,
        num_frames: int,
        use_dmd: bool,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """Generate video using Lyra2 model."""
        if self.current_image is None or self.current_depth is None:
            return None, "Please process a view first"
        
        if not prompt:
            return None, "Please enter a prompt"
        
        try:
            progress(0, desc="Initializing generation...")
            
            # Prepare T5 embeddings
            from lyra_2._src.inference.get_t5_emb import get_umt5_embedding
            desired_dtype = self.model.tensor_kwargs.get("dtype", None)
            desired_device = self.model.tensor_kwargs.get("device", None)
            
            t5 = get_umt5_embedding(prompt, device=desired_device).to(dtype=desired_dtype)
            if t5.dim() == 2:
                t5 = t5.unsqueeze(0)
            elif t5.dim() == 3 and t5.shape[0] != 1:
                t5 = t5[:1]
            
            neg_t5 = misc.to(self.negative_prompt_data["t5_text_embeddings"], **self.model.tensor_kwargs)
            
            # Build camera trajectory
            from lyra_2._src.inference.camera_traj_utils import build_camera_trajectory
            
            initial_w2c = torch.eye(4, dtype=torch.float32, device=desired_device)
            center_depth = torch.quantile(
                self.current_depth[self.current_depth > 0], 0.25
            ).item()
            
            progress(0.2, desc="Building camera trajectory...")
            w2cs_T_44, Ks_T_33 = build_camera_trajectory(
                initial_w2c,
                self.current_intrinsics.to(initial_w2c),
                center_depth,
                num_frames,
                trajectory,
                direction,
                strength,
            )
            
            # Prepare data batch
            H, W = 480, 832
            img_bchw = torch.from_numpy(
                cv2.resize(self.current_image, (W, H)).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device=desired_device) * 2.0 - 1.0
            
            depth_b_thw = self.current_depth.unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1).to(device=desired_device)
            
            w2cs_b_t_44 = w2cs_T_44.unsqueeze(0).to(dtype=torch.float32)
            Ks_b_t_33 = Ks_T_33.unsqueeze(0).to(dtype=torch.float32)
            
            data_batch = {
                "video": img_bchw.unsqueeze(2),
                "t5_text_embeddings": t5,
                "neg_t5_text_embeddings": neg_t5,
                "fps": torch.tensor([16], dtype=torch.int32, device=desired_device),
                "padding_mask": torch.zeros((1, 1, H, W), dtype=self.model.tensor_kwargs["dtype"], device=desired_device),
                "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=desired_device),
                "camera_w2c": w2cs_b_t_44,
                "intrinsics": Ks_b_t_33,
                "depth": depth_b_thw,
            }
            
            progress(0.4, desc="Running video generation...")
            
            # Run generation
            from lyra_2._src.inference.lyra2_ar_inference import run_lyra2_sample, save_output
            
            class Args:
                def __init__(self):
                    self.num_frames = num_frames
                    self.guidance = 5.0
                    self.shift = 5.0
                    self.num_sampling_step = 4 if use_dmd else 50
                    self.use_dmd_scheduler = use_dmd
                    self.seed = 1
                    self.fps = 16
                    self.offload = False
                    self.offload_when_prompt = False
                    self.warp_chunk_size = None
                    self.num_retrieval_views = 1
                    self.disable_cache_update = False
                    self.da3_frame_interval = 8
                    self.da3_max_history_frames = 10
                    self.da3_include_ar_chunk_last_frames = False
                    self.da3_use_predicted_pose = False
                    self.da3_predicted_pose_continuation = False
                    self.offload_da3_diffusion = False
            
            args = Args()
            
            result = run_lyra2_sample(
                self.model,
                data_batch,
                args,
                process_group=None,
                da3_model=self.da3_model,
                show_progress=True,
                log_prefix="streetview_generation",
            )
            
            progress(0.8, desc="Saving video...")
            
            # Save video
            output_path = os.path.join(self.output_dir, "streetview_video.mp4")
            to_show = []
            if result.get("warp_video") is not None:
                to_show.append(result["warp_video"])
            to_show.append(result["video"])
            save_output(to_show, output_path)
            
            self.generated_video_path = output_path
            
            progress(1.0, desc="Complete!")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_path, f"Video generated successfully! Saved to {output_path}"
            
        except Exception as e:
            return None, f"Error generating video: {str(e)}"
    
    def reconstruct_3d(self, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """Reconstruct 3D scene from generated video."""
        if self.generated_video_path is None or not os.path.exists(self.generated_video_path):
            return None, "Please generate a video first"
        
        try:
            progress(0, desc="Starting 3D reconstruction...")
            
            # Run VIPE + DA3 + GS reconstruction
            from lyra_2._src.inference.vipe_da3_gs_recon import main as vipe_da3_main
            
            progress(0.2, desc="Running VIPE pose estimation...")
            
            # Create temporary args
            class Args:
                def __init__(self):
                    self.input_video_path = self.generated_video_path
                    self.output_dir = os.path.join(self.output_dir, "reconstruction")
                    self.force = True
                    self.device = str(self.device)
                    self.no_vipe = False
                    self.vipe_overrides = None
                    self.vipe_full_mode = False
                    self.max_frames = 0
                    self.da3_max_frames = 128
                    self.da3_model_name = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
                    self.da3_model_path_custom = "checkpoints/recon/model.pt"
                    self.da3_process_res = None
                    self.da3_process_method = "upper_bound_resize"
                    self.max_resolution = 0
                    self.gs_down_ratio = 2
                    self.gs_scale_extra_multiplier = 1.0
                    self.gs_ply_prune_opacity_percentile = None
                    self.gs_ds_feature_mode = True
                    self.use_da3_render_pose = True
                    self.render_fps = None
                    self.render_chunk_size = 1
            
            # Temporarily modify sys.argv for argparse
            original_argv = sys.argv
            sys.argv = ['vipe_da3_gs_recon']
            
            try:
                vipe_da3_main()
            finally:
                sys.argv = original_argv
            
            progress(1.0, desc="Complete!")
            
            recon_dir = os.path.join(self.output_dir, "reconstruction")
            ply_path = os.path.join(recon_dir, "reconstructed_scene.ply")
            video_path = os.path.join(recon_dir, "gs_trajectory.mp4")
            
            if os.path.exists(ply_path):
                self.reconstructed_scene_path = ply_path
                return video_path if os.path.exists(video_path) else ply_path, f"3D reconstruction complete! PLY saved to {ply_path}"
            else:
                return None, "3D reconstruction failed to produce output"
            
        except Exception as e:
            return None, f"Error in 3D reconstruction: {str(e)}"
    
    def create_interface(self):
        """Create Gradio interface for Street View."""
        
        with gr.Blocks(
            title="Lyra 2.0 Street View - 3D Scene Generation",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {
                    max-width: 100% !important;
                }
                .panorama-viewer {
                    height: 400px !important;
                }
                .panel-section {
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    margin-bottom: 10px;
                }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🗺️ Lyra 2.0 Street View to 3D
            **Convert Google Street View Panoramas to Explorable 3D Scenes**
            """)
            
            with gr.Row():
                # Left Panel - Street View Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Street View Search")
                    
                    with gr.Group():
                        api_key_input = gr.Textbox(
                            label="Google Street View API Key",
                            placeholder="Enter your API key (optional for demo mode)",
                            type="password"
                        )
                        btn_set_api = gr.Button("🔑 Set API Key", variant="secondary")
                        api_status = gr.Textbox(
                            label="API Status",
                            value="Not configured",
                            interactive=False
                        )
                    
                    with gr.Group():
                        location_search = gr.Textbox(
                            label="Location Search",
                            placeholder="e.g., 'Eiffel Tower, Paris' or '48.8584,2.2945'"
                        )
                        btn_search = gr.Button("🔎 Search Location", variant="primary")
                        search_result = gr.Textbox(
                            label="Search Result",
                            interactive=False,
                            lines=2
                        )
                        location_coords = gr.Textbox(
                            label="Location Coordinates",
                            interactive=False,
                            visible=False
                        )
                    
                    with gr.Group():
                        gr.Markdown("**Download Settings**")
                        num_directions = gr.Slider(
                            minimum=4, maximum=16, step=2, value=8,
                            label="Number of Directions (360° coverage)"
                        )
                        fov = gr.Slider(
                            minimum=60, maximum=120, step=10, value=90,
                            label="Field of View (degrees)"
                        )
                        btn_download = gr.Button("📥 Download Views", variant="primary")
                        download_status = gr.Textbox(
                            label="Download Status",
                            interactive=False
                        )
                    
                    with gr.Group():
                        gr.Markdown("**View Selection**")
                        view_gallery = gr.Gallery(
                            label="Downloaded Views",
                            columns=4,
                            height=300,
                            object_fit="contain"
                        )
                        selected_view_idx = gr.Slider(
                            minimum=0, maximum=7, step=1, value=0,
                            label="Select View"
                        )
                        btn_select_view = gr.Button("✅ Select View", variant="secondary")
                        select_status = gr.Textbox(
                            label="Selection Status",
                            interactive=False
                        )
                
                # Center Panel - Viewport & Processing
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ Viewport")
                    
                    with gr.Tabs():
                        with gr.Tab("Selected View"):
                            selected_view_display = gr.Image(label="Selected Street View", height=500)
                            btn_process = gr.Button("⚡ Process View", variant="secondary")
                        
                        with gr.Tab("Depth Map"):
                            depth_map = gr.Image(label="Depth Map", height=500)
                        
                        with gr.Tab("Generated Video"):
                            output_video = gr.Video(label="Generated Video", height=500)
                        
                        with gr.Tab("3D Reconstruction"):
                            recon_output = gr.Video(label="3D Scene Render", height=500)
                    
                    with gr.Group():
                        gr.Markdown("**Video Generation Settings**")
                        with gr.Row():
                            trajectory = gr.Dropdown(
                                choices=list(CAMERA_TRAJECTORY_CHOICES),
                                value="horizontal_zoom",
                                label="Camera Trajectory"
                            )
                            direction = gr.Radio(
                                choices=["left", "right", "up", "down"],
                                value="right",
                                label="Direction"
                            )
                        with gr.Row():
                            strength = gr.Slider(
                                minimum=0.1, maximum=2.0, step=0.1, value=0.5,
                                label="Trajectory Strength"
                            )
                            num_frames = gr.Slider(
                                minimum=81, maximum=481, step=80, value=161,
                                label="Generate Length (frames)"
                            )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the scene expansion...",
                            lines=2
                        )
                        use_dmd = gr.Checkbox(
                            label="Use DMD Fast Inference (4-step, ~15x faster)",
                            value=False
                        )
                        
                        with gr.Row():
                            btn_generate = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                            btn_reconstruct = gr.Button("🧊 Reconstruct 3D Scene", variant="primary", size="lg")
                    
                    progress_bar = gr.Progress()
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False
                    )
                
                # Right Panel - Scene Info & Export
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Scene Information")
                    
                    with gr.Group():
                        gr.Markdown("**Location Details**")
                        location_info = gr.Textbox(
                            label="Location",
                            value="Not selected",
                            interactive=False
                        )
                        panorama_id = gr.Textbox(
                            label="Panorama ID",
                            value="N/A",
                            interactive=False
                        )
                        num_views_downloaded = gr.Number(
                            label="Views Downloaded",
                            value=0,
                            interactive=False
                        )
                    
                    with gr.Group():
                        gr.Markdown("**Generation Info**")
                        selected_trajectory = gr.Textbox(
                            label="Current Trajectory",
                            value="N/A",
                            interactive=False
                        )
                        generation_time = gr.Textbox(
                            label="Generation Time",
                            value="N/A",
                            interactive=False
                        )
                        scene_quality = gr.Textbox(
                            label="Estimated Quality",
                            value="N/A",
                            interactive=False
                        )
                    
                    with gr.Group():
                        gr.Markdown("**Export Options**")
                        export_format = gr.Radio(
                            choices=["PLY", "OBJ", "GLTF"],
                            value="PLY",
                            label="Export Format"
                        )
                        btn_export_scene = gr.Button("💾 Export 3D Scene", variant="secondary")
                        btn_export_video = gr.Button("🎥 Export Video", variant="secondary")
                        btn_export_all_views = gr.Button("📸 Export All Views", variant="secondary")
                    
                    with gr.Group():
                        gr.Markdown("**Tips**")
                        gr.Markdown("""
                        - Get API key from: https://developers.google.com/maps/documentation/streetview/get-api-key
                        - Use specific addresses for better results
                        - More directions = better 360 coverage
                        - Lower FOV = more detail per view
                        - Use descriptive prompts for better generation
                        """)
            
            # Model loading section
            with gr.Accordion("⚙️ Model Settings", open=False):
                with gr.Row():
                    btn_load_models = gr.Button("📥 Load Models", variant="primary")
                    model_offload = gr.Checkbox(label="Enable Model Offloading", value=False)
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="Models not loaded",
                        interactive=False
                    )
            
            # Event handlers
            btn_set_api.click(
                fn=self.set_api_key,
                inputs=[api_key_input],
                outputs=[api_status]
            )
            
            btn_search.click(
                fn=self.search_location,
                inputs=[location_search],
                outputs=[search_result, location_coords]
            )
            
            btn_download.click(
                fn=self.download_views,
                inputs=[location_coords, num_directions, fov],
                outputs=[view_gallery, download_status]
            )
            
            btn_select_view.click(
                fn=self.select_view,
                inputs=[selected_view_idx],
                outputs=[selected_view_display, select_status]
            )
            
            btn_process.click(
                fn=self.process_view,
                inputs=[selected_view_display],
                outputs=[depth_map, status_text]
            )
            
            btn_generate.click(
                fn=self.generate_video,
                inputs=[
                    prompt, trajectory, direction, strength, 
                    num_frames, use_dmd
                ],
                outputs=[output_video, status_text]
            )
            
            btn_reconstruct.click(
                fn=self.reconstruct_3d,
                outputs=[recon_output, status_text]
            )
            
            btn_load_models.click(
                fn=lambda offload: self.load_models(use_dmd=False, offload=offload),
                inputs=[model_offload],
                outputs=[model_status]
            )
            
            # Placeholder handlers
            btn_export_scene.click(
                fn=lambda: "Export not implemented in this version",
                outputs=[status_text]
            )
            btn_export_video.click(
                fn=lambda: "Video export not implemented in this version",
                outputs=[status_text]
            )
            btn_export_all_views.click(
                fn=lambda: "Views export not implemented in this version",
                outputs=[status_text]
            )
        
        return interface


def main():
    """Launch the Street View GUI."""
    gui = StreetViewGUI()
    interface = gui.create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
