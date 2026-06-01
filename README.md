# ConDiF: Confidence-guided Direction Fields for Structure-aware Diffusion Inpainting

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1+cu116-ee4c2c.svg)](https://pytorch.org/)
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

## Inpainting animation (Indoor)

将 GIF 放在 `docs/images/` 下，用相对路径嵌入即可（GitHub 会自动播放）：

```markdown
![ConDiF indoor inpainting demo](docs/images/indoor_inpaint_demo.gif)
```

当前仓库中的演示：

<p align="center">
  <img src="docs/images/indoor_inpaint_demo.gif" width="70%" alt="ConDiF indoor inpainting: masked input to restored output"/>
</p>
<p align="center">
  <em>Figure 3. Indoor scene inpainting — masked input → ConDiF output.</em>
</p>

<p align="center">
  <img src="docs/images/indoor_damaged.png" width="32%" alt="Masked input"/>
  &nbsp;
  <img src="docs/images/indoor_restored.png" width="32%" alt="ConDiF output"/>
</p>
<p align="center"><em>Left: masked input · Right: ConDiF inpainting result</em></p>

**替换为你自己的 GIF：** 把文件复制为 `docs/images/indoor_inpaint_demo.gif`（或修改上方 `src=` 路径），然后 `git add docs/images/indoor_inpaint_demo.gif` 一并提交。

可选：用脚本从两张 PNG 生成循环 GIF（需 Node.js）：

```bash
node scripts/make_inpaint_gif.cjs docs/images/indoor_damaged.png docs/images/indoor_restored.png docs/images/indoor_inpaint_demo.gif
```

---

## Repository structure

```
Condif2/
├── README.md
├── requirements.txt
├── docs/
│   ├── paper.pdf
│   └── images/
│       ├── overview.png
│       ├── celeba_results.png
│       ├── indoor_damaged.png
│       ├── indoor_restored.png
│       └── indoor_inpaint_demo.gif
├── demo/                          # sample images, masks, precomputed priors
├── src/diffusers/                 # ConDiF model & inference pipeline
│   ├── models/condif.py
│   └── pipelines/condif/pipeline_condif.py
├── examples/condif/
│   └── train_condif_indoor.py     # training entry
└── scripts/
    ├── check_condif_imports.py
    └── make_inpaint_gif.cjs         # build before/after demo GIF (needs Node.js)
```

---

## Installation

Tested with **Python 3.8**, **PyTorch 1.12.1 (CUDA 11.6)**, and the pinned packages in [`requirements.txt`](requirements.txt).

> 本项目使用仓库内 `src/diffusers/`，**不要** `pip install diffusers` 覆盖本地 ConDiF 代码。

```bash
git clone https://github.com/<your-username>/Condif2.git
cd Condif2

conda create -n condif python=3.8 -y
conda activate condif

# 1) PyTorch 1.12.1 + CUDA 11.6（须先于 requirements.txt 安装）
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 \
  --index-url https://download.pytorch.org/whl/cu116

# 2) 其余依赖（见 requirements.txt）
pip install -r requirements.txt

# 3) 训练脚本额外依赖
pip install imgaug webdataset
```

Use the **local** ConDiF package:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python scripts/check_condif_imports.py
```

Expected output: `OK: top-level import`.

Windows (PowerShell):

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
python scripts/check_condif_imports.py
```

<details>
<summary>Full dependency list (<code>requirements.txt</code>)</summary>

```
accelerate==1.0.1
albumentations==1.4.18
clip==0.2.0
controlnet-aux==0.0.10
datasets==3.1.0
einops==0.8.1
huggingface-hub==0.36.2
imageio==2.35.1
kornia==0.7.3
lpips==0.1.4
matplotlib==3.4.3
numpy==1.24.3
opencv-python==4.11.0.86
pandas==2.0.3
Pillow==9.5.0
pytorch-fid==0.3.0
safetensors==0.5.3
scikit-image==0.21.0
scipy==1.10.1
tensorboard==2.13.0
timm==0.6.13
torch==1.12.1+cu116
torchaudio==0.12.1+cu116
torchmetrics==1.2.1
torchvision==0.13.1+cu116
tqdm==4.67.1
transformers==4.46.3
triton==3.0.0
```

</details>

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
