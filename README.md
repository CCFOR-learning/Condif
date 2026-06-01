# ConDiF: Confidence-guided Direction Fields for Structure-aware Diffusion Inpainting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

Official implementation of **ConDiF** — a confidence-guided direction field framework for structure-aware diffusion inpainting on **indoor scenes** and **face images**.

> **Paper:** *ConDiF: Confidence-guided Direction Fields for Structure-aware Diffusion Inpainting*  
> Chen Cheng, Wenkang Jia\*, Qiufeng Wang, Jieda Wei, Jiannan Chen  
> \*Corresponding author: jiawenkang@fjnu.edu.cn

📄 [Paper (PDF)](docs/paper.pdf)

---

## Release status

| Component | Status | Location |
|-----------|--------|----------|
| **ConDiF model & inference pipeline** | ✅ Released | `src/diffusers/` |
| **Training code** | ✅ Released | `examples/condif/train_condif_indoor.py` |
| **Demo samples** (images, masks, precomputed `.npz`) | ✅ Released | `demo/` |
| **SPPM structure prediction** (skeleton → confidence + direction fields) | 🔜 Coming soon | — |

> **Note:** Training and inference expect **precomputed structure priors** (`.npz` with keys `S`, `Dx`, `Dy`). Sample files are in `demo/`. The SPPM module that generates these priors from corrupted images will be released in a future update.

---

## Abstract

Indoor and face inpainting require strong geometric continuity under large masks. **ConDiF** uses a **decoupled dual-branch** design: a frozen Stable Diffusion UNet for generation and a trainable **structure guidance branch** for non-intrusive prior injection. Discrete line sketches are converted into **continuous confidence fields** \(S\) and **tangent direction fields** \((T_x, T_y)\). **FiLM modulation** and **mask gating** apply structure guidance only inside corrupted regions with adaptive strength. Experiments on **Indoor** (ShanghaiTech + NYUDepthV2) and **CelebA-HQ** show improvements over LaMa, ZITS, and related baselines.

---

## Method overview

<p align="center">
  <img src="docs/images/overview.png" width="95%" alt="Structure-Aware Dual-Branch Latent Diffusion Inpainting Module"/>
</p>
<p align="center"><em>Figure 1. Structure-Aware Dual-Branch Latent Diffusion Inpainting Module — frozen UNet backbone + trainable structure guidance branch (FiLM &amp; mask gating, zero-init injection).</em></p>

**Training:** use high-quality priors from **complete** images.  
**Inference:** SPPM (or precomputed NPZ) from **corrupted** images → end-to-end inpainting.

---

## Visual results (CelebA-HQ)

<p align="center">
  <img src="docs/images/celeba_results.png" width="95%" alt="CelebA-HQ face inpainting results"/>
</p>
<p align="center">
  <em>Figure 2. CelebA-HQ face inpainting — Top: masked input · Middle: SPPM structure prediction · Bottom: ConDiF output.</em>
</p>

---

## Repository structure

```
Condif2/
├── README.md
├── docs/
│   ├── paper.pdf
│   └── images/
│       ├── overview.png
│       └── celeba_results.png
├── demo/                          # sample images, masks, precomputed priors
├── src/diffusers/                 # ConDiF model & inference pipeline
│   ├── models/condif.py
│   └── pipelines/condif/pipeline_condif.py
├── examples/condif/
│   └── train_condif_indoor.py     # training entry
└── scripts/
    └── check_condif_imports.py
```

---

## Installation

```bash
git clone https://github.com/<your-username>/Condif2.git
cd Condif2

conda create -n condif python=3.10 -y
conda activate condif

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors opencv-python pillow \
            numpy tqdm scipy webdataset imgaug torchmetrics datasets huggingface_hub
# optional: xformers, wandb, bitsandbytes
```

Use the **local** ConDiF package (do not rely on pip `diffusers` alone):

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python scripts/check_condif_imports.py
```

Expected output: `OK: top-level import`.

---

## Quick start (inference)

Sample assets are in `demo/`. Checkpoints are saved under the `condif/` subfolder.

```python
import os, sys
import torch
import cv2
from PIL import Image

project_root = "/path/to/Condif2"
sys.path.insert(0, os.path.join(project_root, "src"))

from diffusers import (
    CondifModel,
    StableDiffusionCondifPipeline,
    DPMSolverMultistepScheduler,
)

base_model = "runwayml/stable-diffusion-v1-5"
condif_ckpt = "path/to/checkpoint/condif"  # or output_dir/condif after training

condif = CondifModel.from_pretrained(condif_ckpt, torch_dtype=torch.float16)
pipe = StableDiffusionCondifPipeline.from_pretrained(
    base_model,
    condif=condif,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

init_bgr = cv2.imread("demo/indoor_sample.png")
mask = (cv2.imread("demo/indoor_sample_mask.png").sum(-1) > 255)[:, :, None]
init_bgr = init_bgr * (1 - mask)
init_image = Image.fromarray(init_bgr.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray((mask.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

out = pipe(
    prompt="a photo of an indoor scene",
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=50,
    condif_conditioning_scale=1.0,
    skeleton_npz_paths=["demo/indoor_sample.npz"],
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
out.save("result.png")
```

---

## Structure priors (`.npz` format)

Each training / inference sample needs a precomputed `.npz`:

| NPZ key | Meaning |
|---------|---------|
| `S` | Structure confidence field |
| `Dx`, `Dy` | Direction components (pipeline derives tangents `Tx`, `Ty`) |

Fusion weight **α** (paper Eq. 1): `α=0.6` for indoor scenes, `α=0` for CelebA-HQ faces.

---

## Training

Prepare JSONL metadata and a mask pool, then launch training:

**Train JSONL** (one JSON object per line):

```json
{"image": "/path/to/image.png", "skeleton_npz": "/path/to/prior.npz", "prompt": "optional caption"}
```

**Val JSONL** (fixed mask per sample):

```json
{"image": "/path/to/image.png", "mask": "/path/to/mask.png", "skeleton_npz": "/path/to/prior.npz", "prompt": ""}
```

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

accelerate launch examples/condif/train_condif_indoor.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir /path/to/train.jsonl \
  --val_data_dir /path/to/val.jsonl \
  --mask_file_list /path/to/mask_list.txt \
  --output_dir outputs/condif-indoor \
  --resolution 512 \
  --train_batch_size 4 \
  --learning_rate 3e-5 \
  --max_train_steps 20000 \
  --checkpointing_steps 500 \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention
```

| Argument | Description |
|----------|-------------|
| `--pretrained_model_name_or_path` | SD 1.5 base (UNet + VAE + text encoder) |
| `--condif_model_name_or_path` | Optional; init ConDiF from an existing checkpoint |
| `--train_data_dir` | Training JSONL path |
| `--val_data_dir` | Validation JSONL path |
| `--mask_file_list` | Text file listing mask paths (one per line) for training |
| `--output_dir` | Checkpoints saved to `output_dir/condif/` |
| `--resume_from_checkpoint` | `latest` or a step folder name |

Only the **ConDiF structure guidance branch** is trained; UNet, VAE, and text encoder stay frozen.

---

## SPPM structure prediction *(coming soon)*

The **SPPM** module (skeleton extraction → confidence field \(S\) + tangent direction fields \((T_x, T_y)\)) is described in §3.2 of the paper. Code for running SPPM on corrupted images at test time will be released separately.

Until then, use precomputed `.npz` files in `demo/` or your own priors following the format above.

---

## Citation

```bibtex
@article{chen2026condif,
  title   = {ConDiF: Confidence-guided Direction Fields for Structure-aware Diffusion Inpainting},
  author  = {Chen, Cheng and Jia, Wenkang and Wang, Qiufeng and Wei, Jieda and Chen, Jiannan},
  journal = {},
  year    = {2026}
}
```

---

## Acknowledgements

- Built on [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) and a trimmed [Hugging Face Diffusers](https://github.com/huggingface/diffusers) codebase.
- Baseline comparisons include LaMa, ZITS, CTSDG, and other methods cited in the paper.

---

## License

Code in `src/diffusers/` follows the **Apache 2.0** license (Hugging Face Diffusers).  
Model weights derived from Stable Diffusion are subject to the [Stable Diffusion license](https://huggingface.co/runwayml/stable-diffusion-v1-5).

---

## FAQ

**Q: Which parts of the code can I run today?**  
Inference (`src/diffusers/`), training (`examples/condif/`), and demo samples (`demo/`). SPPM prior generation is not yet released.

**Q: `cannot import name 'CondifModel'`**  
Ensure `PYTHONPATH` points to `Condif2/src` first, then run `python scripts/check_condif_imports.py`.

**Q: How do I get structure priors without SPPM?**  
Use precomputed `.npz` files (see `demo/`). Each file should contain `S`, `Dx`, and `Dy`.
