import copy
import csv
import hashlib
import json
import os
import pathlib
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

project_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_path)

threestudio_path = os.path.join(project_path, "extern/threestudio")
sys.path.insert(0, threestudio_path)

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from mesh.util import load_mesh, write_obj_with_texture
from dataset.mesh_dataset import MeshDataset
from models.avatar_image_generator import (
    AvatarImageGenerator,
    to_character_sheet,
    from_character_sheet,
)

from optimization.optimizer3d import Optimizer3D
from optimization.setup_geometry3d import setup_geometry3d
from utils.write_video import (
    write_360_video_diffrast,
    render_with_rotate_light,
)

import threestudio

from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.models.materials.no_material import NoMaterial
from threestudio.models.materials.pbr_material import PBRMaterial
from threestudio.models.background.solid_color_background import SolidColorBackground


REQUIRED_BATCH_CAPTION_FIELDS = ("caption_short", "caption_long")
DEFAULT_SINGLE_PROMPT = "a mouse pirate, detailed, hd"


def parse_args(arglist=None):
    expand_path = lambda p: pathlib.Path(p).expanduser().absolute() if p is not None else None
    parser = ArgumentParser(description="FlashTex")
    parser.add_argument("--input_mesh", type=expand_path, help="Path to input mesh")
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=expand_path,
        default="./output",
        help="Path to output directory",
    )
    parser.add_argument("--production", action="store_true", help="Run in production mode, skipping debug outputs")
    parser.add_argument("--model_id", type=str, default="Lykon/DreamShaper", help="Diffusers model to use for generation")
    parser.add_argument("--controlnet_name", type=str, default="", help="ControlNet model to use for generation")
    parser.add_argument("--pretrained_dir", type=str, help="Directory containing pretrained weights for models")
    parser.add_argument("--distilled_encoder", type=str, default="load/encoder_resnet4.pth", help="Disilled encoder checkpoint")
    parser.add_argument("--image_resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--num_sds_iterations", type=int, default=400, help="Number of iterations for SDS optimization")
    parser.add_argument("--rotation_x", type=float, default=0.0, help="Mesh rotation about the X axis")
    parser.add_argument("--rotation_y", type=float, default=0.0, help="Mesh rotation about the Y axis")
    parser.add_argument("--gif_resolution", type=int, default=512, help="Resolution of spin-around gif")
    parser.add_argument("--refine", action="store_true", help="Refine original mesh texture")
    parser.add_argument("--bbox_size", type=float, default=-1, help="Size of a mesh bbox enclosing mesh avatar/object")
    parser.add_argument("--texture_tile_size", type=int, default=1024, help="Size each texture tile in UV space")
    parser.add_argument("--uv_unwrap", action="store_true", help="Perform uv unwrapping")
    parser.add_argument("--uv_rescale", action="store_true", help="Perform uv rescaling")

    # Arguments for generating the reference image
    parser.add_argument("--disable_img2img", action="store_true", help="Do not use img2img for the character sheet")
    parser.add_argument("--ddim_steps", type=int, default=20, help="DDIM steps")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--img2img_strength", type=float, default=1.0, help="Strength for img2img for the character sheet")
    parser.add_argument("--character_sheet_noise", type=float, default=0.0, help="Character sheet noise scale")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength")
    parser.add_argument("--scale", type=float, default=9.0, help="Scale")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta")

    parser.add_argument("--camera_dist", type=float, default=5.0, help="Camera distance")
    parser.add_argument("--camera_fov", type=float, default=30.0, help="Camera FOV")
    parser.add_argument("--walkaround_y", type=float, default=0.0, help="Walkaround camera Y")
    parser.add_argument("--skip_character_sheet", action="store_true", help="Set to skip character sheet for multiview consistency")
    parser.add_argument("--prompt_masking", dest="prompt_masking_style", type=str, default="global", help="global | front_back | front_back_localized")
    parser.add_argument("--prompt", type=str, default=DEFAULT_SINGLE_PROMPT, help="Text prompt for stable diffusion")
    parser.add_argument("--additional_prompt", dest="a_prompt", type=str, default="", help="Additional text prompt for stable diffusion")
    parser.add_argument("--negative_prompt", dest="n_prompt", type=str, default="bad quality, blurred, low resolution, low quality, low res", help="Negative text prompt for stable diffusion")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu or cuda), defaults to cuda when available.")
    parser.add_argument("--guidance_scale", type=float, default=50.0, help="Guidance Scale")
    parser.add_argument("--cond_strength", type=float, default=1.0, help="Condtioning Strength for ControlNet")
    parser.add_argument("--guidance_sds", type=str, default="SDS_sd", help="Choose from [SDS_sd, SDS_LightControlNet]")
    parser.add_argument("--no_tqdm", action="store_true", help="No tqdm logging")
    parser.add_argument("--SDS_camera_dist", type=float, default=5.0)
    parser.add_argument("--pbr_material", action="store_true", help="Use PBR Material.")
    parser.add_argument("--lambda_recon_reg", type=float, default=1000.0, help="Reconstruction regularization")
    parser.add_argument("--lambda_albedo_smooth", type=float, default=0.0, help="Albedo smoothness regularization")

    # Batch inference options
    parser.add_argument("--tsv_path", type=expand_path, help="Path to batch TSV with obj_id / mesh / caption columns")
    parser.add_argument("--caption_field", type=str, default="caption_long", help="TSV caption field used in batch mode")
    parser.add_argument("--result_tsv", type=expand_path, default=None, help="Path to generated manifest TSV")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of batch samples to process; -1 means all")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples that already have standardized outputs")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop the batch immediately if any sample fails")
    parser.add_argument("--gpu_ids", "--gpu-ids", type=str, default="0", help="Comma-separated physical GPU ids for batch inference")
    parser.add_argument("--num_gpus", "--num-gpus", type=int, default=1, help="Number of GPUs to use from gpu_ids; <=0 means all")
    parser.add_argument("--workers_per_gpu", "--workers-per-gpu", type=str, default="1", help="Parallel workers per GPU, or 'auto'")
    parser.add_argument("--obj_id", type=str, default="", help="Optional object id for single-sample outputs")
    parser.add_argument("--texture_name", type=str, default="", help="Optional texture/output sample name")

    args = parser.parse_args(args=arglist)
    return args


def view_angle_to_prompt(elev, azim):
    azim = azim % 360
    if abs(azim - 180.0) < 90.0:
        return "rear view"
    if abs(azim) < 30.0 or abs(azim - 360) < 30:
        return "front view"
    return "side view"


def args_to_dict(args):
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            result[key] = str(value.absolute())
        else:
            result[key] = value
    return result


def compute_effective_sample_seed(args):
    base_seed = int(getattr(args, "seed", 0))
    mesh_path = ""
    if getattr(args, "input_mesh", None) is not None:
        mesh_path = os.path.abspath(str(args.input_mesh))

    prompt = str(getattr(args, "prompt", "") or "")
    seed_key = "\n".join(
        [
            str(base_seed),
            mesh_path,
            prompt,
            str(getattr(args, "model_id", "") or ""),
            str(getattr(args, "controlnet_name", "") or ""),
            str(getattr(args, "guidance_sds", "") or ""),
            str(getattr(args, "rotation_x", 0.0)),
            str(getattr(args, "rotation_y", 0.0)),
            str(getattr(args, "num_sds_iterations", 0)),
            str(bool(getattr(args, "pbr_material", False))),
        ]
    )
    seed_hash = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(seed_hash[:4], "big")


def configure_deterministic_torch():
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        print(f"[WARN] Failed to enable deterministic algorithms: {exc}")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup(args):
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def resolve_result_tsv(args):
    if args.result_tsv is None:
        return os.path.abspath(os.path.join(str(args.output_dir), "generated_manifest.tsv"))

    result_tsv = str(args.result_tsv)
    if not os.path.isabs(result_tsv):
        result_tsv = os.path.join(str(args.output_dir), result_tsv)
    return os.path.abspath(result_tsv)


def load_batch_from_tsv(tsv_path, caption_field):
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []

        missing = [name for name in REQUIRED_BATCH_CAPTION_FIELDS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"TSV missing required caption columns: {', '.join(missing)} "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )
        if caption_field not in fieldnames:
            raise ValueError(
                f"TSV missing caption field '{caption_field}' "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )
        if "mesh" not in fieldnames:
            raise ValueError(
                f"TSV missing required mesh column "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )

        return [row for row in reader]


def compose_batch_prompt(row, args):
    caption = (row.get(args.caption_field) or "").strip()
    if caption:
        return caption
    return (args.prompt or "").strip()


def get_view_params(mode="character_sheet", num_views=2):
    if mode == "character_sheet":
        if num_views == 2:
            elev = torch.tensor([0.0, 0.0])
            azim = torch.tensor([0.0, 180.0])
            light_dirs = [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        elif num_views == 4:
            elev = torch.tensor([0.0, 0.0, 15.0, 15.0])
            azim = torch.tensor([0.0, 180.0, -75.0, 75.0])
            light_dirs = [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        else:
            raise NotImplementedError(f"Unsupported number of views {num_views}")
    else:
        raise NotImplementedError(f"Unsupported view mode {mode}")

    return elev, azim, light_dirs


def get_mesh_dict_and_view_strings(args, use_textures=False):
    mesh_dataset = MeshDataset(
        input_mesh=args.input_mesh,
        device=args.device,
        head_only=False,
        texture_tile_size=args.texture_tile_size,
        bbox_size=args.bbox_size,
        rotation_x=args.rotation_x,
        rotation_y=args.rotation_y,
        uv_unwrap=args.uv_unwrap,
        uv_rescale=args.uv_rescale,
        use_existing_textures=use_textures,
    )

    elev, azim, light_dirs = get_view_params(num_views=4)

    mesh_dict = mesh_dataset.render_images(
        image_resolution=args.image_resolution,
        dist=torch.tensor(args.camera_dist),
        elev=elev,
        azim=azim,
        fov=torch.tensor(args.camera_fov),
        light_dirs=light_dirs,
        use_textures=use_textures,
    )

    view_strings = [view_angle_to_prompt(elev[i], azim[i]) for i in range(mesh_dict["mesh_images"].size(0))]

    if not args.production:
        torchvision.utils.save_image(mesh_dict["mesh_images"], f"{args.output_dir}/projected_mesh.png", padding=0)

    return mesh_dict, view_strings


def generate_avatar_image(args, mesh_dict, view_strings, use_view_prompt, prompt_masking_style, is_character_sheet, input_images, output_name=""):
    output_image_name = f"{args.output_dir}/{output_name}"

    avatar_image_generator = AvatarImageGenerator(args, preloaded_models=None, device=args.device)
    generated_outputs = avatar_image_generator(
        mesh_dict,
        view_strings=view_strings,
        use_view_prompt=use_view_prompt,
        prompt_masking_style=prompt_masking_style,
        input_images=input_images,
        is_character_sheet=is_character_sheet,
        img2img_strength=args.img2img_strength,
    )
    output_images = generated_outputs["images"]

    if not args.production and output_name:
        torchvision.utils.save_image(output_images, output_image_name, padding=0)

    return output_images, generated_outputs.get("diffusion_noise_init", None)


def interpolate_image(image0, image1, t):
    return (image0 * t) + (image1 * (1.0 - t))


def generate_initial_character_sheet(args, mesh_dict, view_strings):
    if args.character_sheet_noise > 0.0:
        noise_mask = (mesh_dict["mesh_depths"] > 0.05).float()
        input_noise = (torch.randn_like(mesh_dict["mesh_images"]) * noise_mask) * args.character_sheet_noise
        mesh_dict["mesh_images"] = torch.clamp(
            interpolate_image(mesh_dict["mesh_images"], input_noise, 1.0 - args.character_sheet_noise),
            0.0,
            1.0,
        )

    return generate_avatar_image(
        args=args,
        mesh_dict=mesh_dict,
        view_strings=view_strings,
        use_view_prompt=args.skip_character_sheet,
        prompt_masking_style=args.prompt_masking_style,
        is_character_sheet=not args.skip_character_sheet,
        input_images=mesh_dict["mesh_images"],
        output_name="depth2image.png",
    )


def setup_renderers(tsdf, use_pbr=False, bg_color=(0.0, 0.0, 0.0), device="cuda", bg_random_p=0.5):
    material = PBRMaterial(
        {
            "min_albedo": 0.03,
            "max_albedo": 0.8,
        }
    ).to(device) if use_pbr else NoMaterial({}).to(device)

    bg = SolidColorBackground(dict(color=bg_color, random_aug=False, hls_color=True, s_range=(0.0, 0.01), random_aug_prob=bg_random_p)).to(device)
    bg_test = SolidColorBackground(dict(color=bg_color)).to(device)
    optimization_renderer = NVDiffRasterizer({"context_type": "cuda"}, geometry=tsdf, background=bg, material=material)
    test_renderer = NVDiffRasterizer({"context_type": "cuda"}, geometry=tsdf, background=bg_test, material=material)
    return dict(optimization=optimization_renderer, testing=test_renderer)


def copy_if_exists(src, dst):
    if not os.path.isfile(src):
        return ""
    shutil.copyfile(src, dst)
    return os.path.abspath(dst)


def rewrite_file_with_replacements(src, dst, replacements):
    if not os.path.isfile(src):
        return ""

    with open(src, "r", encoding="utf-8") as f:
        content = f.read()

    for old, new in replacements.items():
        content = content.replace(old, new)

    with open(dst, "w", encoding="utf-8") as f:
        f.write(content)

    return os.path.abspath(dst)


def export_obj_without_mtl(src, dst, rotate_z_180=False):
    if not os.path.isfile(src):
        return ""

    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("mtllib ") or stripped.startswith("usemtl "):
            continue
        if rotate_z_180 and (stripped.startswith("v ") or stripped.startswith("vn ")):
            prefix, coords = stripped.split(None, 1)
            parts = coords.strip().split()
            if len(parts) >= 3:
                x = -float(parts[0])
                y = -float(parts[1])
                z = float(parts[2])
                suffix = "".join(f" {value}" for value in parts[3:])
                line = f"{prefix} {x} {y} {z}{suffix}\n"
        filtered_lines.append(line)

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

    return os.path.abspath(dst)


def prune_sample_output_dir(output_dir):
    keep_files = {"mesh.obj", "albedo.png", "normal.png", "metallic.png", "roughness.png"}

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            continue
        if name not in keep_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def save_standardized_outputs(output_dir):
    optimization_output_dir = os.path.join(output_dir, "optimization")

    copy_if_exists(os.path.join(optimization_output_dir, "texture_kd.png"), os.path.join(output_dir, "albedo.png"))
    copy_if_exists(os.path.join(optimization_output_dir, "texture_nrm.png"), os.path.join(output_dir, "normal.png"))
    copy_if_exists(os.path.join(optimization_output_dir, "texture_metallic.png"), os.path.join(output_dir, "metallic.png"))
    copy_if_exists(os.path.join(optimization_output_dir, "texture_roughness.png"), os.path.join(output_dir, "roughness.png"))

    obj_src = os.path.join(optimization_output_dir, "output_mesh.obj")
    obj_dst = os.path.join(output_dir, "mesh.obj")

    export_obj_without_mtl(obj_src, obj_dst, rotate_z_180=True)
    prune_sample_output_dir(output_dir)


def direct_optimization_nvdiffrast(args, mesh_dict, target_images, target_masks, progress_callback=None):
    textured_mesh = mesh_dict["mesh"]

    output_mesh_basename = "output_mesh.obj"
    output_texture_basename = "tex_combined.png"
    tmp_mesh_dir = tempfile.mkdtemp(prefix="tmp_mesh_")
    tmp_mesh_filename = os.path.join(tmp_mesh_dir, output_mesh_basename)
    print("tmp_mesh_filename", tmp_mesh_filename)
    write_obj_with_texture(tmp_mesh_filename, output_texture_basename, textured_mesh)
    try:
        iter_num = args.num_sds_iterations
        guidance = args.guidance_sds

        implicit3d = setup_geometry3d(
            mesh_file=tmp_mesh_filename,
            geometry="custom_mesh",
            centering="none",
            scaling="none",
            material="pbr" if args.pbr_material else "no_material",
        )

        renderers = setup_renderers(implicit3d, use_pbr=args.pbr_material, bg_random_p=1.0)

        optimization_output_dir = os.path.join(args.output_dir, "optimization")
        os.makedirs(optimization_output_dir, exist_ok=True)

        optimizer3d = Optimizer3D(
            tsdf=implicit3d,
            renderers=renderers,
            model_name=args.model_id,
            controlnet_name=args.controlnet_name,
            output_dir=optimization_output_dir,
            distilled_encoder=args.distilled_encoder,
            lambda_recon_reg=args.lambda_recon_reg,
            lambda_albedo_smooth=args.lambda_albedo_smooth,
            grad_clip=0.1,
            save_img=0 if args.production else 100,
            save_video=0 if args.production else 1000,
            fix_geometry=True,
            pretrained_dir=args.pretrained_dir,
            guidance=guidance,
            guidance_scale=args.guidance_scale,
            cond_strength=args.cond_strength,
            no_tqdm=args.no_tqdm,
            camera_dist=args.SDS_camera_dist,
        )

        optimizer3d.optimize_with_prompts(
            prompt=args.prompt,
            negative_prompt=args.n_prompt,
            num_iters=iter_num,
            textured_mesh=textured_mesh,
            fixed_target_images=target_images,
            fixed_target_masks=F.interpolate(target_masks, size=(512, 512), mode="bilinear"),
            fixed_target_azim=mesh_dict["azim"],
            fixed_target_elev=mesh_dict["elev"],
            progress_callback=progress_callback,
        )

        if not args.production:
            write_360_video_diffrast(renderers["testing"], output_filename=f"{optimization_output_dir}/{guidance}_final_rgb.gif")
            render_with_rotate_light(renderers["optimization"], output_filename=f"{optimization_output_dir}/{guidance}_final_rgb_rotate.gif")
            write_360_video_diffrast(renderers["testing"], output_filename=f"{optimization_output_dir}/{guidance}_final_rgb_up.gif", elev=-30)
            shutil.copyfile(f"{optimization_output_dir}/{guidance}_final_rgb.gif", f"{args.output_dir}/video360.gif")

        optimizer3d.export_mesh(
            optimization_output_dir,
            textured_mesh.textures.verts_uvs_padded().squeeze(0),
            textured_mesh.textures.faces_uvs_padded().squeeze(0),
        )
        save_standardized_outputs(args.output_dir)
    finally:
        shutil.rmtree(tmp_mesh_dir, ignore_errors=True)


def run_single_inference(args, progress_callback=None):
    args = setup(args)
    sample_seed = compute_effective_sample_seed(args)
    configure_deterministic_torch()
    seed_everything(sample_seed)
    args.seed = sample_seed
    print(f"[INFO] Effective sample seed: {sample_seed}")

    mesh_dict, view_strings = get_mesh_dict_and_view_strings(args, use_textures=args.refine)

    if args.guidance_sds == "SDS_sd":
        output_images, _ = generate_initial_character_sheet(args, mesh_dict, view_strings)
        output_images = F.interpolate(output_images, size=(512, 512), mode="bilinear")
        torchvision.utils.save_image(output_images[0:1], f"{args.output_dir}/depth2image_front.png", padding=0)
    else:
        output_images = None

    direct_optimization_nvdiffrast(args, mesh_dict, output_images, mesh_dict["mesh_masks"], progress_callback=progress_callback)


def path_if_exists(sample_dir, name):
    path = os.path.join(sample_dir, name)
    return os.path.abspath(path) if os.path.exists(path) else ""


def build_result_row(obj_id, sample_dir, caption_short=None, caption_long=None, caption_used=None):
    sample_dir = os.path.abspath(sample_dir)
    row = {
        "obj_id": obj_id,
        "mesh": path_if_exists(sample_dir, "mesh.obj"),
        "albedo": path_if_exists(sample_dir, "albedo.png"),
        "rough": path_if_exists(sample_dir, "roughness.png"),
        "metal": path_if_exists(sample_dir, "metallic.png"),
        "normal": path_if_exists(sample_dir, "normal.png"),
    }
    if caption_short is not None:
        row["caption_short"] = caption_short
    if caption_long is not None:
        row["caption_long"] = caption_long
    if caption_used is not None:
        row["caption_used"] = caption_used
    return row


def append_to_manifest(tsv_path, new_rows):
    if not new_rows:
        return

    existing_rows = []
    existing_ids = set()
    if os.path.isfile(tsv_path):
        with open(tsv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                obj_id = row.get("obj_id", "")
                existing_rows.append(row)
                existing_ids.add(obj_id)

    rows_to_add = [row for row in new_rows if row["obj_id"] not in existing_ids]
    all_rows = existing_rows + rows_to_add

    fieldnames = ["obj_id", "mesh", "albedo", "rough", "metal", "normal"]
    for name in ("caption_short", "caption_long", "caption_used"):
        if any(name in row for row in all_rows):
            fieldnames.append(name)

    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"[INFO] Updated generated manifest: {tsv_path} (total rows: {len(all_rows)})")


def save_experiment_config(exp_dir, args, processed_samples, skipped_samples=None, manifest_path=None, timing_info=None):
    cfg = {
        "options": args_to_dict(args),
        "tsv_path": os.path.abspath(str(args.tsv_path)) if args.tsv_path else None,
        "processed_samples": processed_samples,
    }
    if skipped_samples:
        cfg["skipped_samples"] = skipped_samples
    if manifest_path:
        cfg["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        cfg["timing"] = timing_info

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def merge_and_save_experiment_config(exp_dir, args, new_processed, new_skipped=None, manifest_path=None, timing_info=None):
    config_path = os.path.join(exp_dir, "config.json")

    if not os.path.isfile(config_path):
        save_experiment_config(exp_dir, args, new_processed, new_skipped, manifest_path, timing_info)
        return

    with open(config_path, "r", encoding="utf-8") as f:
        existing_config = json.load(f)

    prev_processed = existing_config.get("processed_samples", [])
    processed_by_id = {sample["obj_id"]: sample for sample in prev_processed}
    for sample in new_processed:
        processed_by_id[sample["obj_id"]] = sample
    merged_processed = [processed_by_id[key] for key in sorted(processed_by_id.keys())]

    prev_skipped = existing_config.get("skipped_samples", [])
    merged_skipped = {sample["obj_id"]: sample for sample in prev_skipped}
    processed_ids = set(processed_by_id.keys())
    for obj_id in list(merged_skipped.keys()):
        if obj_id in processed_ids:
            del merged_skipped[obj_id]
    for sample in new_skipped or []:
        if sample["obj_id"] not in processed_ids:
            merged_skipped[sample["obj_id"]] = sample

    timing_key = "timing"
    timing_index = 2
    while timing_key in existing_config:
        timing_key = f"timing{timing_index}"
        timing_index += 1

    existing_config["options"] = args_to_dict(args)
    existing_config["tsv_path"] = os.path.abspath(str(args.tsv_path)) if args.tsv_path else None
    existing_config["processed_samples"] = merged_processed
    existing_config["skipped_samples"] = [merged_skipped[key] for key in sorted(merged_skipped.keys())]
    if manifest_path:
        existing_config["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        existing_config[timing_key] = timing_info

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)

    print(
        f"[INFO] Updated config.json: {len(merged_processed)} processed, "
        f"{len(existing_config['skipped_samples'])} skipped, timing key: {timing_key}"
    )


def check_sample_completed(sample_dir, require_pbr_assets):
    required_files = ["mesh.obj", "albedo.png"]
    if require_pbr_assets:
        required_files.extend(["normal.png", "metallic.png", "roughness.png"])
    return all(os.path.isfile(os.path.join(sample_dir, name)) for name in required_files)


def make_progress_callback(output_dir):
    def progress_callback(image):
        torchvision.utils.save_image(image, os.path.join(output_dir, "progress.png"), padding=0)

    return progress_callback


def print_timing_summary(timing_info):
    print("\n" + "=" * 60)
    print("[TIMING] Inference completed!")
    print(f"[TIMING] Start time: {timing_info['start_time']}")
    print(f"[TIMING] End time: {timing_info['end_time']}")
    print(
        f"[TIMING] Total time: {timing_info['total_time_formatted']} "
        f"({timing_info['total_seconds']:.2f} seconds)"
    )
    print(f"[TIMING] Newly processed samples: {timing_info['num_samples_processed']}")
    if "num_samples_reused" in timing_info:
        print(f"[TIMING] Reused existing samples: {timing_info['num_samples_reused']}")
    if "num_samples_skipped" in timing_info:
        print(f"[TIMING] Failed / skipped samples: {timing_info['num_samples_skipped']}")
    print("=" * 60 + "\n")


def finalize_timing(start_time, start_datetime, num_processed, num_reused=0, num_skipped=0):
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_seconds = end_time - start_time
    total_time_formatted = str(timedelta(seconds=int(total_seconds)))
    avg_seconds = total_seconds / num_processed if num_processed > 0 else 0.0

    timing_info = {
        "start_time": start_datetime,
        "end_time": end_datetime,
        "total_seconds": round(total_seconds, 2),
        "total_time_formatted": total_time_formatted,
        "num_samples_processed": num_processed,
        "avg_seconds_per_sample": round(avg_seconds, 2),
        "num_samples_reused": num_reused,
        "num_samples_skipped": num_skipped,
    }
    print_timing_summary(timing_info)
    return timing_info


def parse_gpu_ids(gpu_ids_str):
    gpu_ids_str = str(gpu_ids_str).strip()
    if gpu_ids_str.startswith("[") and gpu_ids_str.endswith("]"):
        gpu_ids_str = gpu_ids_str[1:-1]
    return [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]


def estimate_workers_per_gpu(gpu_id, model_memory_gb=12.0, safety_margin=0.85):
    try:
        if not torch.cuda.is_available():
            return 1, 0

        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        available_gb = total_memory_gb * safety_margin
        estimated = int(available_gb / model_memory_gb)
        workers = max(1, min(estimated, 4))
        return workers, total_memory_gb
    except Exception as exc:
        print(f"[WARN] Failed to query GPU {gpu_id} memory: {exc}")
        return 1, 0


def calculate_workers_per_gpu(gpu_ids, workers_per_gpu_str):
    workers_per_gpu_str = str(workers_per_gpu_str).strip().lower()

    if workers_per_gpu_str == "auto":
        if not gpu_ids:
            return 1
        workers, total_mem = estimate_workers_per_gpu(gpu_ids[0])
        print(f"[INFO] Auto-detected GPU memory: {total_mem:.1f} GB")
        print(f"[INFO] Auto-calculated workers_per_gpu: {workers}")
        return workers

    try:
        return max(1, int(workers_per_gpu_str))
    except ValueError:
        print(f"[WARN] Invalid workers_per_gpu '{workers_per_gpu_str}', using default 1")
        return 1


def create_sample_args(base_args, mesh_path, sample_output_dir, obj_id, prompt):
    sample_args = copy.deepcopy(base_args)
    sample_args.tsv_path = None
    sample_args.input_mesh = pathlib.Path(mesh_path)
    sample_args.output_dir = pathlib.Path(sample_output_dir)
    sample_args.prompt = prompt
    sample_args.obj_id = obj_id
    sample_args.texture_name = obj_id
    sample_args.result_tsv = None
    return sample_args


def run_single_gpu_worker(gpu_id, worker_id, rows_subset, base_args, tsv_dir, textures_dir):
    worker_tag = f"[GPU {gpu_id} W{worker_id}]"

    if torch.cuda.is_available():
        print(f"{worker_tag} CUDA available, device count: {torch.cuda.device_count()}")
        print(f"{worker_tag} Current device: {torch.cuda.current_device()}")
        print(f"{worker_tag} Device name: {torch.cuda.get_device_name(0)}")

    processed_samples = []
    skipped_samples = []

    for local_idx, (global_idx, row) in enumerate(rows_subset):
        obj_id = (row.get("obj_id") or "").strip() or f"sample_{global_idx}"
        mesh_path = (row.get("mesh") or "").strip()
        caption_short = (row.get("caption_short") or "").strip()
        caption_long = (row.get("caption_long") or "").strip()
        prompt = compose_batch_prompt(row, base_args)

        if not mesh_path or not prompt:
            print(f"{worker_tag} Skip row {global_idx}: missing mesh or prompt (obj_id={obj_id})")
            skipped_samples.append({"obj_id": obj_id, "reason": "missing mesh or prompt"})
            continue

        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(tsv_dir, mesh_path)
        mesh_path = os.path.abspath(mesh_path)

        sample_output_dir = os.path.join(textures_dir, obj_id)
        if base_args.skip_existing and check_sample_completed(sample_output_dir, require_pbr_assets=base_args.pbr_material):
            print(f"{worker_tag} Reusing existing sample {obj_id} ({local_idx + 1}/{len(rows_subset)}, global {global_idx + 1})")
            processed_samples.append(
                build_result_row(
                    obj_id,
                    sample_output_dir,
                    caption_short=caption_short,
                    caption_long=caption_long,
                    caption_used=base_args.caption_field,
                )
            )
            continue

        sample_args = create_sample_args(base_args, mesh_path, sample_output_dir, obj_id, prompt)
        print(f"{worker_tag} Processing {obj_id} ({local_idx + 1}/{len(rows_subset)}, global {global_idx + 1})")

        try:
            run_single_inference(sample_args, progress_callback=make_progress_callback(sample_output_dir))
            processed_samples.append(
                build_result_row(
                    obj_id,
                    sample_output_dir,
                    caption_short=caption_short,
                    caption_long=caption_long,
                    caption_used=base_args.caption_field,
                )
            )
        except Exception as exc:
            print(f"{worker_tag} Error processing {obj_id}: {exc}")
            traceback.print_exc()
            skipped_samples.append({"obj_id": obj_id, "reason": str(exc)})
            if base_args.stop_on_error:
                raise

    return processed_samples, skipped_samples


def worker_subprocess_entry():
    gpu_id = int(os.environ["FLASHTEX_GPU_ID"])
    worker_id = int(os.environ.get("FLASHTEX_WORKER_ID", "0"))
    config_file = os.environ["FLASHTEX_CONFIG_FILE"]

    worker_tag = f"[GPU {gpu_id} W{worker_id}]"
    print(f"{worker_tag} Starting subprocess...")
    print(f"{worker_tag} CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    with open(config_file, "rb") as f:
        config = pickle.load(f)

    base_args = Namespace(**config["args_dict"])
    rows_subset = config["rows_subset"]
    tsv_dir = config["tsv_dir"]
    textures_dir = config["textures_dir"]
    result_file = config["result_file"]

    processed_samples, skipped_samples = run_single_gpu_worker(
        gpu_id,
        worker_id,
        rows_subset,
        base_args,
        tsv_dir,
        textures_dir,
    )

    with open(result_file, "wb") as f:
        pickle.dump(
            {
                "gpu_id": gpu_id,
                "worker_id": worker_id,
                "processed": processed_samples,
                "skipped": skipped_samples,
            },
            f,
        )

    print(f"{worker_tag} Finished. Processed {len(processed_samples)}, skipped {len(skipped_samples)}")


def run_multi_gpu(base_args, indexed_rows, tsv_dir, textures_dir, gpu_ids, workers_per_gpu=1):
    num_gpus = len(gpu_ids)
    total_workers = num_gpus * workers_per_gpu
    worker_assignments = [[] for _ in range(total_workers)]

    for idx, indexed_row in enumerate(indexed_rows):
        worker_assignments[idx % total_workers].append(indexed_row)

    print(
        f"[INFO] Distributing {len(indexed_rows)} samples across "
        f"{num_gpus} GPUs x {workers_per_gpu} workers = {total_workers} total workers"
    )
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        gpu_total = sum(len(worker_assignments[gpu_idx * workers_per_gpu + w]) for w in range(workers_per_gpu))
        print(f"  GPU {gpu_id}: {gpu_total} samples ({workers_per_gpu} workers)")

    temp_dir = tempfile.mkdtemp(prefix="flashtex_multiGPU_")
    print(f"[INFO] Using temp directory: {temp_dir}")

    args_dict = args_to_dict(base_args)
    processes = []
    result_files = []

    try:
        for gpu_idx, gpu_id in enumerate(gpu_ids):
            for local_worker_id in range(workers_per_gpu):
                global_worker_id = gpu_idx * workers_per_gpu + local_worker_id
                rows_subset = worker_assignments[global_worker_id]
                if not rows_subset:
                    continue

                config_file = os.path.join(temp_dir, f"config_gpu{gpu_id}_worker{local_worker_id}.pkl")
                result_file = os.path.join(temp_dir, f"result_gpu{gpu_id}_worker{local_worker_id}.pkl")
                result_files.append((gpu_id, local_worker_id, result_file))

                with open(config_file, "wb") as f:
                    pickle.dump(
                        {
                            "args_dict": args_dict,
                            "rows_subset": rows_subset,
                            "tsv_dir": tsv_dir,
                            "textures_dir": textures_dir,
                            "result_file": result_file,
                        },
                        f,
                    )

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["FLASHTEX_GPU_ID"] = str(gpu_id)
                env["FLASHTEX_WORKER_ID"] = str(local_worker_id)
                env["FLASHTEX_CONFIG_FILE"] = config_file

                cmd = [
                    sys.executable,
                    "-c",
                    "from generate_texture import worker_subprocess_entry; worker_subprocess_entry()",
                ]

                print(f"[INFO] Launching subprocess for GPU {gpu_id} Worker {local_worker_id} ({len(rows_subset)} samples)...")
                processes.append(
                    (
                        gpu_id,
                        local_worker_id,
                        subprocess.Popen(
                            cmd,
                            env=env,
                            cwd=os.path.dirname(os.path.abspath(__file__)),
                            stdout=None,
                            stderr=None,
                        ),
                    )
                )

        print(f"[INFO] Waiting for {len(processes)} subprocesses to complete...")
        worker_failures = []
        for gpu_id, local_worker_id, process in processes:
            return_code = process.wait()
            if return_code != 0:
                worker_failures.append((gpu_id, local_worker_id, return_code))
                print(f"[WARN] Subprocess for GPU {gpu_id} Worker {local_worker_id} exited with code {return_code}")

        all_processed = []
        all_skipped = []
        for gpu_id, local_worker_id, result_file in result_files:
            if not os.path.exists(result_file):
                print(f"[WARN] Result file missing for GPU {gpu_id} Worker {local_worker_id}: {result_file}")
                continue
            with open(result_file, "rb") as f:
                results = pickle.load(f)
            all_processed.extend(results["processed"])
            all_skipped.extend(results["skipped"])
            print(
                f"[INFO] Collected results from GPU {gpu_id} Worker {local_worker_id}: "
                f"{len(results['processed'])} processed, {len(results['skipped'])} skipped"
            )

        if worker_failures and base_args.stop_on_error:
            raise RuntimeError(f"One or more worker subprocesses failed: {worker_failures}")

        all_processed.sort(key=lambda x: x["obj_id"])
        all_skipped.sort(key=lambda x: x["obj_id"])
        return all_processed, all_skipped
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_batch(args):
    output_root = os.path.abspath(str(args.output_dir))
    os.makedirs(output_root, exist_ok=True)
    textures_dir = os.path.join(output_root, "textures")
    os.makedirs(textures_dir, exist_ok=True)

    result_tsv_path = resolve_result_tsv(args)
    args.result_tsv = pathlib.Path(result_tsv_path)

    tsv_path = os.path.abspath(str(args.tsv_path))
    batch_rows = load_batch_from_tsv(tsv_path, args.caption_field)
    total_rows = len(batch_rows)
    if args.max_samples > 0:
        batch_rows = batch_rows[:args.max_samples]

    print(f"[INFO] Loaded {total_rows} rows from {tsv_path}")
    print(f"[INFO] Processing {len(batch_rows)} rows with caption field '{args.caption_field}'")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Result manifest: {result_tsv_path}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("No valid GPU ids provided via --gpu_ids")
    num_gpus = min(args.num_gpus, len(gpu_ids)) if args.num_gpus > 0 else len(gpu_ids)
    gpu_ids = gpu_ids[:num_gpus]
    workers_per_gpu = calculate_workers_per_gpu(gpu_ids, args.workers_per_gpu)
    print(f"[INFO] Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    print(f"[INFO] Workers per GPU: {workers_per_gpu}")

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing_samples = []
    skipped_samples = []
    rows_to_process = []
    tsv_dir = os.path.dirname(tsv_path)

    for idx, row in enumerate(batch_rows):
        obj_id = (row.get("obj_id") or "").strip() or f"sample_{idx}"
        mesh_path = (row.get("mesh") or "").strip()
        caption_short = (row.get("caption_short") or "").strip()
        caption_long = (row.get("caption_long") or "").strip()
        prompt = compose_batch_prompt(row, args)

        if not mesh_path or not prompt:
            reason = "missing mesh or prompt"
            print(f"[WARN] Skip row {idx} ({obj_id}): {reason}")
            skipped_samples.append({"obj_id": obj_id, "reason": reason})
            continue

        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(tsv_dir, mesh_path)
        mesh_path = os.path.abspath(mesh_path)

        sample_output_dir = os.path.join(textures_dir, obj_id)
        if args.skip_existing and check_sample_completed(sample_output_dir, require_pbr_assets=args.pbr_material):
            print(f"[INFO] Reusing existing sample {obj_id} ({idx + 1}/{len(batch_rows)})")
            existing_samples.append(
                build_result_row(
                    obj_id,
                    sample_output_dir,
                    caption_short=caption_short,
                    caption_long=caption_long,
                    caption_used=args.caption_field,
                )
            )
            continue

        rows_to_process.append((idx, row))

    if rows_to_process:
        processed_samples, worker_skipped_samples = run_multi_gpu(
            args,
            rows_to_process,
            tsv_dir,
            textures_dir,
            gpu_ids,
            workers_per_gpu,
        )
        skipped_samples.extend(worker_skipped_samples)
    else:
        processed_samples = []

    available_samples = existing_samples + processed_samples
    if available_samples:
        append_to_manifest(result_tsv_path, available_samples)

    timing_info = finalize_timing(
        start_time,
        start_datetime,
        num_processed=len(processed_samples),
        num_reused=len(existing_samples),
        num_skipped=len(skipped_samples),
    )
    merge_and_save_experiment_config(
        output_root,
        args,
        available_samples,
        new_skipped=skipped_samples,
        manifest_path=result_tsv_path if available_samples else None,
        timing_info=timing_info,
    )


def run_single_cli(args):
    if args.input_mesh is None:
        raise ValueError("--input_mesh is required when --tsv_path is not provided")

    output_root = os.path.abspath(str(args.output_dir))
    os.makedirs(output_root, exist_ok=True)

    obj_id = args.obj_id.strip() or args.texture_name.strip()
    if not obj_id:
        obj_id = pathlib.Path(args.input_mesh).stem
    args.obj_id = obj_id
    args.texture_name = obj_id

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_single_inference(args, progress_callback=make_progress_callback(output_root))

    finalize_timing(start_time, start_datetime, num_processed=1)


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.tsv_path:
        run_batch(cli_args)
    else:
        run_single_cli(cli_args)
