# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lyra 2.0 Interactive GUI
Professional interface for explorable generative 3D world creation.
"""

import os
import sys
import gc
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import threading
import queue

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lyra_2._src.inference.camera_traj_utils import CAMERA_TRAJECTORY_CHOICES
from lyra_2._src.utils.model_loader import load_model_from_checkpoint
from lyra_2._src.inference.depth_utils import load_da3_model, load_moge_model
from lyra_2._ext.imaginaire.utils import log, misc


class LyraGUI:
    """Main GUI class for Lyra 2.0 interactive scene generation."""
    
    def __init__(self):
        self.model = None
        self.da3_model = None
        self.moge_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = "checkpoints/model"
        self.output_dir = "outputs/gui"
        self.current_image = None
        self.current_depth = None
        self.current_intrinsics = None
        self.camera_trajectory = None
        self.generated_video_path = None
        self.reconstructed_scene_path = None
        self.progress_queue = queue.Queue()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            
            # Load MoGe model for scale alignment
            self.moge_model = load_moge_model(self.device)
            self.moge_model.eval()
            status += f"✓ MoGe model loaded for depth alignment\n"
            
            status += "\n✓ All models loaded successfully!"
            return status
            
        except Exception as e:
            return f"Error loading models: {str(e)}"
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process input image and estimate depth."""
        if image is None:
            return None, "Please upload an image first."
        
        try:
            self.current_image = image
            
            # Convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Estimate depth using DA3
            H, W = 480, 832  # Target resolution
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
            
            return depth_vis, f"Image processed successfully. Resolution: {H}x{W}"
            
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
    
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
            return None, "Please process an image first."
        
        if not prompt:
            return None, "Please enter a prompt."
        
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
                log_prefix="gui_generation",
            )
            
            progress(0.8, desc="Saving video...")
            
            # Save video
            output_path = os.path.join(self.output_dir, "generated_video.mp4")
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
            return None, "Please generate a video first."
        
        try:
            progress(0, desc="Starting 3D reconstruction...")
            
            # Run VIPE + DA3 + GS reconstruction
            from lyra_2._src.inference.vipe_da3_gs_recon import main as vipe_da3_main
            import argparse
            
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
                return None, "3D reconstruction failed to produce output."
            
        except Exception as e:
            return None, f"Error in 3D reconstruction: {str(e)}"
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get current camera information."""
        if self.current_depth is None:
            return {
                "Position": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                "Rotation": {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0},
                "Status": "No image loaded"
            }
        
        center_depth = torch.quantile(self.current_depth[self.current_depth > 0], 0.5).item()
        
        return {
            "Position": {"X": 0.0, "Y": 0.0, "Z": center_depth},
            "Rotation": {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0},
            "Status": "Ready"
        }
    
    def create_interface(self):
        """Create Gradio interface."""
        
        with gr.Blocks(
            title="Lyra 2.0 - Explorable Generative 3D Worlds",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {
                    max-width: 100% !important;
                }
                .main-viewport {
                    height: 600px !important;
                }
                .panel-section {
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    margin-bottom: 10px;
                }
                .status-available {
                    color: #28a745;
                    font-weight: bold;
                }
                .status-busy {
                    color: #dc3545;
                    font-weight: bold;
                }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🎬 Lyra 2.0 - Explorable Generative 3D Worlds
            **Professional Interface for Interactive Scene Generation**
            """)
            
            with gr.Row():
                # Left Panel - Camera Controls & Video Generation
                with gr.Column(scale=1):
                    gr.Markdown("### 🎥 Camera Path & Video Generator")
                    
                    with gr.Group():
                        gr.Markdown("**Camera Path Details**")
                        cam_x = gr.Number(label="X", value=0.0, interactive=False)
                        cam_y = gr.Number(label="Y", value=0.0, interactive=False)
                        cam_z = gr.Number(label="Z", value=0.0, interactive=False)
                        cam_pitch = gr.Number(label="Pitch", value=0.0, interactive=False)
                        cam_yaw = gr.Number(label="Yaw", value=0.0, interactive=False)
                        cam_roll = gr.Number(label="Roll", value=0.0, interactive=False)
                        
                        with gr.Row():
                            btn_record_path = gr.Button("🔴 Record Camera Path", variant="secondary")
                            btn_clear_path = gr.Button("🗑️ Clear Path", variant="secondary")
                            btn_smooth_path = gr.Button("🌊 Smooth Path", variant="secondary")
                    
                    # Camera View Preview
                    camera_view = gr.Image(label="Camera View Preview", height=200)
                    
                    with gr.Group():
                        gr.Markdown("**Video Generation**")
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
                        strength = gr.Slider(
                            minimum=0.1, maximum=2.0, step=0.1, value=0.5,
                            label="Trajectory Strength"
                        )
                        num_frames = gr.Slider(
                            minimum=81, maximum=481, step=80, value=161,
                            label="Generate Length (frames)"
                        )
                        prompt_type = gr.Radio(
                            choices=["Text", "Image"],
                            value="Text",
                            label="Prompt Type"
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the scene you want to generate...",
                            lines=3
                        )
                        use_dmd = gr.Checkbox(
                            label="Use DMD Fast Inference (4-step, ~15x faster)",
                            value=False
                        )
                    
                    with gr.Group():
                        progress_bar = gr.Progress()
                        status_text = gr.Textbox(
                            label="Status",
                            value="Model not loaded",
                            interactive=False
                        )
                        btn_generate = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                        btn_reconstruct = gr.Button("🧊 Reconstruct 3D Scene", variant="primary", size="lg")
                
                # Center Panel - Main Viewport
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ Main Viewport")
                    
                    with gr.Tabs():
                        with gr.Tab("Input Image"):
                            input_image = gr.Image(label="Upload Input Image", height=500)
                            btn_process = gr.Button("⚡ Process Image", variant="secondary")
                        
                        with gr.Tab("Depth Map"):
                            depth_map = gr.Image(label="Depth Map", height=500)
                        
                        with gr.Tab("Generated Video"):
                            output_video = gr.Video(label="Generated Video", height=500)
                        
                        with gr.Tab("3D Reconstruction"):
                            recon_output = gr.Video(label="3D Scene Render", height=500)
                    
                    with gr.Group():
                        gr.Markdown("**Viewport Controls**")
                        with gr.Row():
                            btn_reset_view = gr.Button("🔄 Reset View", size="sm")
                            btn_fit_view = gr.Button("📐 Fit to View", size="sm")
                            btn_toggle_grid = gr.Button("🔲 Toggle Grid", size="sm")
                
                # Right Panel - Hierarchy & Properties
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Scene Hierarchy")
                    
                    hierarchy_tree = gr.Tree(
                        label="Scene Structure",
                        value=[
                            {"label": "Main Cloud", "children": [
                                {"label": "RGB Layer"},
                                {"label": "Depth Map"},
                                {"label": "Normals"}
                            ]},
                            {"label": "Camera Trajectory", "children": [
                                {"label": "Keyframes"},
                                {"label": "Frustum"}
                            ]}
                        ]
                    )
                    
                    with gr.Group():
                        gr.Markdown("**Layer Properties**")
                        layer_visibility = gr.CheckboxGroup(
                            choices=["RGB", "Depth", "Normals", "Trajectory"],
                            value=["RGB", "Trajectory"],
                            label="Visible Layers"
                        )
                        layer_opacity = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.1, value=1.0,
                            label="Layer Opacity"
                        )
                    
                    with gr.Group():
                        gr.Markdown("**Scene Statistics**")
                        num_points = gr.Number(label="Number of Points", value=0, interactive=False)
                        scene_bounds = gr.Textbox(
                            label="Scene Bounds",
                            value="Not computed",
                            interactive=False
                        )
                        memory_usage = gr.Textbox(
                            label="GPU Memory",
                            value="Not available",
                            interactive=False
                        )
                    
                    with gr.Group():
                        gr.Markdown("**Export Options**")
                        export_format = gr.Radio(
                            choices=["PLY", "OBJ", "GLTF"],
                            value="PLY",
                            label="Export Format"
                        )
                        btn_export = gr.Button("💾 Export Scene", variant="secondary")
                        btn_export_video = gr.Button("🎥 Export Video", variant="secondary")
            
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
            btn_load_models.click(
                fn=lambda offload: self.load_models(use_dmd=False, offload=offload),
                inputs=[model_offload],
                outputs=[model_status]
            )
            
            btn_process.click(
                fn=self.process_image,
                inputs=[input_image],
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
            
            # Placeholder handlers for UI elements
            btn_record_path.click(
                fn=lambda: "Recording not implemented in this version",
                outputs=[status_text]
            )
            btn_clear_path.click(
                fn=lambda: "Path cleared",
                outputs=[status_text]
            )
            btn_smooth_path.click(
                fn=lambda: "Path smoothed",
                outputs=[status_text]
            )
            
            btn_reset_view.click(
                fn=lambda: "View reset",
                outputs=[status_text]
            )
            btn_fit_view.click(
                fn=lambda: "View fitted",
                outputs=[status_text]
            )
            btn_toggle_grid.click(
                fn=lambda: "Grid toggled",
                outputs=[status_text]
            )
            
            btn_export.click(
                fn=lambda: "Export not implemented in this version",
                outputs=[status_text]
            )
            btn_export_video.click(
                fn=lambda: "Video export not implemented in this version",
                outputs=[status_text]
            )
        
        return interface


def main():
    """Launch the GUI."""
    gui = LyraGUI()
    interface = gui.create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
