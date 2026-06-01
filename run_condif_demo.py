# ConDiF inpainting demo (structure-prior fusion, no metric evaluation).
import os
import sys
import warnings

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import traceback
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

local_diffusers_path = os.path.join(project_root, "src")
if local_diffusers_path not in sys.path:
    sys.path.insert(0, local_diffusers_path)

from diffusers import CondifModel, DPMSolverMultistepScheduler, StableDiffusionCondifPipeline


def _build_inpainting_pipeline(base_model_path, guidance_model, torch_dtype):
    return StableDiffusionCondifPipeline.from_pretrained(
        base_model_path,
        condif=guidance_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )


def _resize_and_align(img, target_size, is_mask=False):
    if img.size == target_size:
        return img
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize(target_size, interp)


def _get_damaged_mask(mask_pil, mask_white_is_damaged=True):
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    if mask_white_is_damaged:
        damaged_mask = (mask_np > 0.5).astype(np.float32)
    else:
        damaged_mask = (mask_np <= 0.5).astype(np.float32)
    return damaged_mask


def _resolve_path(path):
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(project_root, path.lstrip("/\\"))
    return candidate


def _load_image(path, mode="RGB", min_size=16):
    if not path:
        return None
    full_path = _resolve_path(path)
    if full_path and os.path.exists(full_path):
        try:
            img = Image.open(full_path).convert(mode)
            if img.width < min_size or img.height < min_size:
                print(f"Warning: image too small ({img.width}x{img.height}): {full_path}")
                return None
            return img
        except Exception as exc:
            print(f"Warning: failed to load {full_path}: {exc}")
            return None
    print(f"Warning: image not found: {path}")
    return None


def load_skeleton_npz(npz_path):
    data = np.load(npz_path)
    s = data["S"].astype(np.float32)
    dx = data["Dx"].astype(np.float32)
    dy = data["Dy"].astype(np.float32)
    tx = -dy
    ty = dx
    mag = np.sqrt(tx**2 + ty**2)
    mag = np.where(mag < 1e-6, 1.0, mag)
    tx /= mag
    ty /= mag
    return s, tx, ty


def direction_to_rgb(s, tx, ty):
    h, w = s.shape
    angle = np.arctan2(ty, tx) + np.pi
    hue = angle / (2 * np.pi)
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = 1.0
    hsv[..., 2] = s
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
    return rgb


def match_color(source_img, target_img, mask_np):
    non_mask_region = mask_np < 0.5
    if np.sum(non_mask_region) < 100:
        return source_img

    matched_img = source_img.copy().astype(np.float32)
    for channel in range(3):
        src_channel = source_img[..., channel].astype(np.float32)
        tgt_channel = target_img[..., channel].astype(np.float32)
        src_mean = np.mean(src_channel[non_mask_region])
        src_std = np.std(src_channel[non_mask_region])
        tgt_mean = np.mean(tgt_channel[non_mask_region])
        tgt_std = np.std(tgt_channel[non_mask_region])
        if src_std < 1e-3:
            src_std = 1e-3
        matched_channel = (src_channel - src_mean) * (tgt_std / src_std) + tgt_mean
        matched_img[..., channel] = np.clip(matched_channel, 0, 255)
    return matched_img.astype(np.uint8)


def guided_filter(guide, src, radius, eps):
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
    mean_guide_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    var_guide = mean_guide_guide - mean_guide * mean_guide
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * guide + mean_b


def adaptive_gaussian_blend(original_img, repaired_img, mask_np):
    original_np = np.array(original_img).astype(np.float32)
    repaired_np = np.array(repaired_img).astype(np.float32)
    mask_area = np.sum(mask_np)
    kernel_size = max(3, min(21, int(math.sqrt(mask_area) / 10) * 2 + 1))
    gray_original = cv2.cvtColor(original_np.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    mask_blurred_raw = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
    mask_blurred = guided_filter(
        gray_original,
        mask_blurred_raw,
        radius=max(1, kernel_size // 2),
        eps=1e-6,
    )
    mask_blurred = mask_blurred[..., np.newaxis]
    blended_img = original_np * (1 - mask_blurred) + repaired_np * mask_blurred
    return blended_img.astype(np.uint8)


def poisson_blend(original_img, repaired_img, mask_pil):
    src = np.array(repaired_img)
    dst = np.array(original_img)
    mask = np.array(mask_pil)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
    y_indices, x_indices = np.where(mask_gray > 127)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return original_img
    center = (int(np.mean(x_indices)), int(np.mean(y_indices)))
    try:
        output = cv2.seamlessClone(src, dst, mask_gray, center, cv2.MIXED_CLONE)
        return Image.fromarray(output)
    except Exception:
        return repaired_img


def blend_repaired_image(original_img, repaired_img, mask_pil, blend_mode="gaussian", enable_color_match=True):
    original_np = np.array(original_img).astype(np.uint8)
    repaired_np = np.array(repaired_img).astype(np.uint8)
    mask_np = _get_damaged_mask(mask_pil)
    if enable_color_match:
        repaired_np = match_color(repaired_np, original_np, mask_np)
        repaired_img = Image.fromarray(repaired_np)
    if blend_mode == "none":
        final_img = repaired_np
    elif blend_mode == "gaussian":
        final_img = adaptive_gaussian_blend(original_img, repaired_img, mask_np)
    elif blend_mode == "poisson":
        final_img = poisson_blend(original_img, repaired_img, mask_pil)
    else:
        print(f"Warning: unknown blend mode '{blend_mode}', using raw output")
        final_img = repaired_np
    if isinstance(final_img, np.ndarray):
        return Image.fromarray(final_img.astype(np.uint8))
    return final_img


def display_results(
    original,
    damaged,
    mask,
    repaired,
    structure_s=None,
    structure_tx=None,
    structure_ty=None,
    sample_id="",
    prompt="",
    blend_mode="",
):
    has_structure = structure_s is not None and structure_tx is not None and structure_ty is not None
    n_cols = 5 if has_structure else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    if n_cols == 1:
        axes = [axes]
    titles = ["Original", "Masked", "Mask", "Repaired"]
    images = [original, damaged, mask, repaired]
    for ax, title, img in zip(axes[:4], titles, images):
        if title == "Mask":
            ax.imshow(np.array(img), cmap="gray")
        else:
            ax.imshow(np.array(img))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    if has_structure:
        rgb_dir = direction_to_rgb(structure_s, structure_tx, structure_ty)
        axes[4].imshow(rgb_dir)
        axes[4].set_title("Structure prior (HSV)", fontsize=12)
        axes[4].axis("off")
    fig.suptitle(f"Sample: {sample_id} | Blend: {blend_mode}", fontsize=14, y=1.05)
    if prompt:
        plt.figtext(0.5, -0.05, f"Prompt: {prompt[:100]}...", ha="center", fontsize=10, wrap=True)
    plt.tight_layout()
    plt.show()


class CondifEvaluator:
    def __init__(self, base_model_path, condif_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.base_model_path = base_model_path
        self.condif_path = condif_path
        self.pipeline = None

    def load_pipeline(self, force_fp16=True):
        print("Loading inpainting pipeline...")
        if not os.path.isabs(self.base_model_path):
            self.base_model_path = os.path.join(project_root, self.base_model_path)
        if not os.path.isabs(self.condif_path):
            self.condif_path = os.path.join(project_root, self.condif_path)
        print(f"   Base model: {self.base_model_path}")
        print(f"   ConDiF branch: {self.condif_path}")
        desired_dtype = torch.float16 if (force_fp16 and torch.cuda.is_available()) else torch.float32
        try:
            guidance_model = CondifModel.from_pretrained(self.condif_path, torch_dtype=desired_dtype)
            self.pipeline = _build_inpainting_pipeline(self.base_model_path, guidance_model, desired_dtype)
            print(f"   Model loaded ({desired_dtype})")
        except Exception as exc:
            print(f"   FP16 load failed, retrying in float32: {exc}")
            guidance_model = CondifModel.from_pretrained(self.condif_path, torch_dtype=torch.float32)
            self.pipeline = _build_inpainting_pipeline(self.base_model_path, guidance_model, torch.float32)
            desired_dtype = torch.float32
            print("   Model loaded in float32")
        try:
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
        except Exception:
            pass
        try:
            self.pipeline.to(self.device)
        except Exception:
            pass
        try:
            if hasattr(self.pipeline, "scheduler"):
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++",
                )
                print("   Scheduler: DPM++ SDE Karras")
        except Exception as exc:
            print(f"   Scheduler setup failed: {exc}")
        print("Pipeline ready.")
        return True

    def run_inference(
        self,
        damaged_img,
        mask_img,
        skeleton_npz_path=None,
        prompt="",
        num_steps=50,
        guidance_scale=1.2,
        conditioning_scale=1.0,
        seed=42,
    ):
        if self.pipeline is None:
            print("Error: pipeline not loaded")
            return None
        if mask_img.mode != "L":
            mask_img = mask_img.convert("L")
        if mask_img.size != damaged_img.size:
            mask_img = mask_img.resize(damaged_img.size, Image.NEAREST)

        base_prompt = prompt if prompt else "a clean indoor room"
        enhanced_prompt = f"RAW photo, {base_prompt}, 8k uhd, dslr, soft lighting, high quality"
        negative_prompt = (
            "text, cropped, out of frame, worst quality, low quality, jpeg artifacts, "
            "ugly, duplicate, blurry, bad anatomy, deformed, disfigured"
        )
        gen_kwargs = {
            "prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "image": damaged_img,
            "mask": mask_img,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "condif_conditioning_scale": conditioning_scale,
            "generator": torch.Generator(device=self.device).manual_seed(seed),
            "output_type": "pil",
        }
        if skeleton_npz_path is not None and os.path.exists(skeleton_npz_path):
            gen_kwargs["skeleton_npz_paths"] = (
                [skeleton_npz_path, skeleton_npz_path] if guidance_scale > 1.0 else [skeleton_npz_path]
            )
            print(f"   Structure NPZ: {os.path.basename(skeleton_npz_path)}")
        else:
            gen_kwargs["skeleton_npz_paths"] = None

        try:
            return self.pipeline(**gen_kwargs).images[0]
        except Exception as exc:
            print(f"Inference failed: {exc}")
            traceback.print_exc()
            return None


def save_comparison_grid(
    original,
    damaged,
    mask,
    repaired,
    save_path,
    structure_s=None,
    structure_tx=None,
    structure_ty=None,
    prompt="",
):
    has_structure = structure_s is not None and structure_tx is not None and structure_ty is not None
    n_cols = 5 if has_structure else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.2, 3.2))
    if n_cols == 1:
        axes = [axes]
    panels = [
        ("Input (GT)", original),
        ("Masked", damaged),
        ("Mask", mask),
        ("ConDiF output", repaired),
    ]
    for ax, (title, img) in zip(axes[:4], panels):
        if title == "Mask":
            ax.imshow(np.array(img), cmap="gray")
        else:
            ax.imshow(np.array(img))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    if has_structure:
        rgb_dir = direction_to_rgb(structure_s, structure_tx, structure_ty)
        axes[4].imshow(rgb_dir)
        axes[4].set_title("Structure prior", fontsize=11)
        axes[4].axis("off")
    subtitle = prompt[:120] + ("..." if prompt and len(prompt) > 120 else "")
    if subtitle.strip():
        fig.suptitle(subtitle, fontsize=10, y=0.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved comparison: {save_path}")


DEMO_SETTINGS = {
    "base_model": "ckpt/Realistic_Vision_V6.0_B1_noVAE",
    "condif_model": "ckpt/condif",
    "output_dir": "demo/output",
    "num_steps": 30,
    "guidance_scale": 3.8,
    "conditioning_scale": 1.1,
    "seed": 42,
    "blend_mode": "gaussian",
    "enable_color_match": True,
    "save_comparison": True,
}

DEMO_SAMPLES = [
    {
        "name": "indoor",
        "image": "demo/indoor_sample.png",
        "mask": "demo/indoor_sample_mask.png",
        "skeleton_npz": "demo/indoor_sample.npz",
        "prompt": "a kitchen with a stove, oven, sink, and cabinets. Indoor scene, high quality, detailed.",
    },
    {
        "name": "celebahq",
        "image": "demo/celebahq_sample.png",
        "mask": "demo/celebahq_sample_mask.png",
        "skeleton_npz": "demo/celebahq_sample.npz",
        "prompt": "a high quality portrait photo of a person, detailed face, natural lighting, photorealistic.",
    },
]


def run_single_sample(sample_cfg, global_cfg, base_model=None, condif_model=None, output_dir=None, show_plot=False):
    cfg = dict(global_cfg)
    cfg.update({k: v for k, v in sample_cfg.items() if k != "name"})
    name = sample_cfg.get("name", "demo")
    base_model = base_model or cfg.get("base_model")
    condif_model = condif_model or cfg.get("condif_model") or global_cfg.get("condif_model")
    out_root = _resolve_path(output_dir or cfg.get("output_dir", "demo/output"))
    sample_out = os.path.join(out_root, name)
    os.makedirs(sample_out, exist_ok=True)

    original_img = _load_image(cfg.get("image"))
    mask_img = _load_image(cfg.get("mask"), mode="L")
    if original_img is None or mask_img is None:
        print(f"Error [{name}]: missing image or mask")
        return False
    if mask_img.size != original_img.size:
        mask_img = mask_img.resize(original_img.size, Image.NEAREST)

    skeleton_npz_path = _resolve_path(cfg.get("skeleton_npz"))
    if skeleton_npz_path and not os.path.isfile(skeleton_npz_path):
        print(f"Warning [{name}]: structure NPZ not found: {skeleton_npz_path}")
        skeleton_npz_path = None

    damaged_np = np.array(original_img).copy()
    mask_np = np.array(mask_img) > 127
    damaged_np[mask_np] = 0
    damaged_img = Image.fromarray(damaged_np)

    evaluator = CondifEvaluator(base_model, condif_model)
    if not evaluator.load_pipeline():
        return False

    repaired_raw = evaluator.run_inference(
        damaged_img,
        mask_img,
        skeleton_npz_path,
        cfg.get("prompt", ""),
        num_steps=int(cfg.get("num_steps", 30)),
        guidance_scale=float(cfg.get("guidance_scale", 3.8)),
        conditioning_scale=float(cfg.get("conditioning_scale", 1.1)),
        seed=int(cfg.get("seed", 42)),
    )
    if repaired_raw is None:
        return False

    repaired_img = blend_repaired_image(
        original_img,
        repaired_raw,
        mask_img,
        cfg.get("blend_mode", "gaussian"),
        bool(cfg.get("enable_color_match", True)),
    )
    repaired_img = _resize_and_align(repaired_img, original_img.size)
    damaged_img.save(os.path.join(sample_out, "damaged.png"))
    repaired_img.save(os.path.join(sample_out, "repaired.png"))
    print(f"   Saved: {os.path.join(sample_out, 'repaired.png')}")

    s = tx = ty = None
    if skeleton_npz_path and os.path.exists(skeleton_npz_path):
        try:
            s, tx, ty = load_skeleton_npz(skeleton_npz_path)
            h, w = repaired_img.height, repaired_img.width
            if s.shape != (h, w):
                s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
                tx = cv2.resize(tx, (w, h), interpolation=cv2.INTER_LINEAR)
                ty = cv2.resize(ty, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass

    if cfg.get("save_comparison", True):
        save_comparison_grid(
            original_img,
            damaged_img,
            mask_img,
            repaired_img,
            os.path.join(sample_out, "comparison.png"),
            structure_s=s,
            structure_tx=tx,
            structure_ty=ty,
            prompt=cfg.get("prompt", ""),
        )

    if show_plot:
        display_results(
            original_img,
            damaged_img,
            mask_img,
            repaired_img,
            s,
            tx,
            ty,
            sample_id=name,
            prompt=cfg.get("prompt", ""),
            blend_mode=cfg.get("blend_mode", "gaussian"),
        )
    return True


def run_demo(base_model=None, condif_model=None, output_dir=None, show_plot=False, sample_name=None):
    global_cfg = dict(DEMO_SETTINGS)
    samples = list(DEMO_SAMPLES)
    if sample_name:
        samples = [s for s in samples if s["name"] == sample_name]
        if not samples:
            print(f"Error: unknown sample '{sample_name}'")
            return False

    print("=" * 70)
    print("ConDiF demo")
    print(f"   Samples: {', '.join(s['name'] for s in samples)}")
    print("=" * 70)

    ok_all = True
    for sample in samples:
        ok = run_single_sample(
            sample,
            global_cfg,
            base_model=base_model,
            condif_model=condif_model,
            output_dir=output_dir,
            show_plot=show_plot,
        )
        if not ok:
            ok_all = False
    print("\nDone." if ok_all else "\nSome samples failed.")
    return ok_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConDiF inpainting demo")
    parser.add_argument("--base-model", default=None, help="SD base model path or HF repo id")
    parser.add_argument("--condif-model", default=None, help="ConDiF checkpoint directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--sample", choices=tuple(s["name"] for s in DEMO_SAMPLES), default=None)
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")
    args = parser.parse_args()

    ok = run_demo(
        base_model=args.base_model,
        condif_model=args.condif_model,
        output_dir=args.output_dir,
        show_plot=args.show,
        sample_name=args.sample,
    )
    if not ok:
        raise SystemExit(1)
