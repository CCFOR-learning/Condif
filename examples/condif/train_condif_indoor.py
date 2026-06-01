import os
import sys
import torch
from torchmetrics import StructuralSimilarityIndexMeasure
# 强制开启PyTorch原生SDPA注意力加速，优先级最高
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
# 1. 获取当前训练脚本的绝对路径
current_script_path = os.path.abspath(__file__) 
# 2. 计算项目根目录 (ConDiF) 的路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 3. 构造本地 diffusers 源码的完整路径
local_diffusers_path = os.path.join(project_root, 'src')
# 4. 将本地路径插入到模块搜索路径的最前面
sys.path.insert(0, local_diffusers_path)
from PIL import ImageEnhance
# ============ 新增代码结束 ============
from collections import defaultdict
import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import json
import cv2
import imgaug.augmenters as iaa
import io
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.models.condif import CondifOutput

import diffusers
from diffusers import (
    AutoencoderKL,
    CondifModel,
    DDPMScheduler,
    StableDiffusionCondifPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import webdataset as wds
import time  # 补充必要导入


def build_inpainting_pipeline(pretrained_model_name_or_path, guidance_model, **kwargs):
    """Create an inpainting pipeline with ConDiF guidance."""
    return StableDiffusionCondifPipeline.from_pretrained(
        pretrained_model_name_or_path,
        condif=guidance_model,
        **kwargs,
    )

# ============ 修复：处理 is_npu_available 导入问题 ============
import accelerate.utils

if not hasattr(accelerate.utils, 'is_npu_available'):
    def is_npu_available():
        return False
    accelerate.utils.is_npu_available = is_npu_available
    
try:
    from accelerate.utils import is_npu_available
except ImportError:
    def is_npu_available():
        return False
    import accelerate.utils
    accelerate.utils.is_npu_available = is_npu_available
# ============ 修复结束 ============
if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")

logger = get_logger(__name__)

class EMAModel:
    def __init__(self, parameters, decay=0.999):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.data.clone().detach() for p in self.parameters]
        self.temp_params = None

    def update(self):
        for shadow, param in zip(self.shadow_params, self.parameters):
            if param.requires_grad:
                new_shadow = (1 - self.decay) * param.data + self.decay * shadow
                shadow.copy_(new_shadow)

    def copy_to(self, target_parameters):
        for target, shadow in zip(target_parameters, self.shadow_params):
            target.data.copy_(shadow.data)

    def store(self, target_parameters):
        self.temp_params = [p.data.clone().detach() for p in target_parameters]

    def restore(self, target_parameters):
        if self.temp_params is None:
            return
        for target, temp in zip(target_parameters, self.temp_params):
            target.data.copy_(temp.data)
        self.temp_params = None
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# ============ 加载 skeleton npz 生成结构先验 ============
def load_skeleton_npz(npz_path):
    data = np.load(npz_path)
    S = data['S'].astype(np.float32)
    Dx = data['Dx'].astype(np.float32)
    Dy = data['Dy'].astype(np.float32)
    Tx = -Dy
    Ty = Dx
    mag = np.sqrt(Tx**2 + Ty**2)
    mag = np.where(mag < 1e-6, 1.0, mag)
    Tx /= mag
    Ty /= mag
    return S, Tx, Ty
# ================================================================

def log_validation(
    vae, text_encoder, tokenizer, unet, condif, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        condif = accelerator.unwrap_model(condif)
    else:
        condif = CondifModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = build_inpainting_pipeline(
        args.pretrained_model_name_or_path,
        condif,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt) and len(args.validation_image) == len(args.validation_mask):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
        validation_masks = args.validation_mask
    else:
        raise ValueError(
            "number of `args.validation_image`, `args.validation_mask`, and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    for validation_prompt, validation_image, validation_mask in zip(validation_prompts, validation_images, validation_masks):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_mask = Image.open(validation_mask).convert("RGB")
        validation_image = Image.composite(Image.new('RGB', (validation_image.size[0], validation_image.size[1]), (0, 0, 0)), validation_image, validation_mask.convert("L"))

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt, validation_image, validation_mask, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="ConDiF conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# ConDiF-{repo_id}

These are ConDiF structure guidance weights trained on {base_model}.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "condif",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="ConDiF structure-aware diffusion inpainting training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--condif_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained condif model or model identifier from huggingface.co/models."
        " If not specified condif weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="condif-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",  # ---------- 修改：余弦退火，跳出局部最优 ----------
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=5000, help="Number of steps for the warmup in the lr scheduler.")  # ---------- 修改：更长 warmup ----------
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay to use.")  # ---------- 修改：降低权重衰减，保留细节 ----------
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    # ---------- 新增验证集参数 ----------
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        help="Path to the validation JSONL file. If provided, validation loss will be computed and logged.",
    )
    # ----------------------------------
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the condif conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=["A cake on the table."],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=["examples/condif/src/test_image.jpg"],
        nargs="+",
        help=(
            "A set of paths to the paintingnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        default=["examples/condif/src/test_mask.jpg"],
        nargs="+",
        help=(
            "A set of paths to the paintingnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_condif",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--random_mask",
        action="store_true",
        help=(
            "Training ConDiF with random mask"
        ),
    )
    parser.add_argument(
        "--mask_file_list",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to mask list file(s) for training (each line is a mask image path). If multiple lists are given, they will be merged and randomly sampled per image each epoch."
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the condif encoder."
        )

    return args
def rotate_image_and_mask(image, damaged_image, mask, S, Tx, Ty, angle):
    """
    对图像、受损图像、mask、结构先验进行相同角度的旋转。
    angle: 旋转角度（度），正值逆时针。
    返回旋转后的所有数组。
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 图像使用双线性插值
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    damaged_image = cv2.warpAffine(damaged_image, M, (w, h), flags=cv2.INTER_LINEAR)
    # mask 使用最近邻插值保持二值性
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    # 结构先验使用双线性插值
    S = cv2.warpAffine(S, M, (w, h), flags=cv2.INTER_LINEAR)
    Tx = cv2.warpAffine(Tx, M, (w, h), flags=cv2.INTER_LINEAR)
    Ty = cv2.warpAffine(Ty, M, (w, h), flags=cv2.INTER_LINEAR)
    return image, damaged_image, mask, S, Tx, Ty

class LocalCondifDataset(Dataset):
    """
    支持训练/验证模式：
    - 训练：从多个mask列表文件随机选择mask，jsonl只需包含image和skeleton_npz。
    - 验证：从jsonl读取固定mask（需包含mask字段）。
    提示词强制为空，以实现与ZITS的公平对比。
    """
    def __init__(self, jsonl_path, tokenizer, resolution=512, random_mask=False,
                 proportion_empty_prompts=0.0, is_train=True, mask_file_list=None):
        super().__init__()
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.random_mask = random_mask  # 保留但不再使用，仅作兼容
        self.proportion_empty_prompts = proportion_empty_prompts  # 强制空提示，此参数无效
        self.is_train = is_train

        # 读取 jsonl
        self.records = []
        with open(jsonl_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # 必要字段检查
                if 'image' not in rec:
                    raise ValueError(f"Missing 'image' field in line: {line}")
                if 'skeleton_npz' not in rec:
                    raise ValueError(f"Missing 'skeleton_npz' field in line: {line}")
                # 验证集需要 mask 字段
                if not is_train and 'mask' not in rec:
                    raise ValueError(f"Validation set requires 'mask' field in line: {line}")
                # 读取 prompt 字段（若没有则默认为空）
                rec['prompt'] = rec.get('prompt', '')
                self.records.append(rec)

        # 训练集：加载掩码池
        self.mask_pool = None
        if is_train and mask_file_list is not None:
            self.mask_pool = []
            for list_file in mask_file_list:
                with open(list_file, 'r') as f:
                    self.mask_pool.extend([line.strip() for line in f if line.strip()])
            print(f"[Train] Loaded {len(self.mask_pool)} masks from {mask_file_list}")

    def __len__(self):
        return len(self.records)

    def tokenize_caption(self, rec):
        # ---------- 修改：合理利用真实 prompt，只做轻微正则化 ----------
        if self.is_train:
            # 90%概率用真实 prompt，10%概率置空（防止过拟合）
            if random.random() < 0.1:
                caption = ""
            else:
                caption = rec.get('prompt', '')
        else:
            # 验证集：直接用真实 prompt
            caption = rec.get('prompt', '')
        # ---------- 修改结束 ----------
        
        if not isinstance(caption, str):
            caption = str(caption)
        inputs = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                                padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids[0]

    def load_img(self, p, mode='rgb'):
        if p is None or p == "":
            return None
        im = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
        if im is None:
            raise RuntimeError(f"Cannot read image: {p}")
        im = im[:, :, ::-1]  # convert BGR->RGB
        return im

    def load_mask(self, p, size=None):
        if p is None or p == "":
            return None
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Cannot read mask: {p}")
        m = (m > 127).astype(np.uint8) * 255
        if size is not None and (m.shape[0] != size[0] or m.shape[1] != size[1]):
            m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return m

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = rec["image"]
        skeleton_npz_path = rec["skeleton_npz"]

        # 加载原图
        image = self.load_img(image_path)
        if image is None:
            raise RuntimeError(f"Cannot load image: {image_path}")
        H, W = image.shape[:2]

        # ---------- 获取掩码 ----------
        if self.is_train:
            # 训练：从掩码池随机选一个
            if self.mask_pool is None:
                raise ValueError("Training requires mask_file_list to be provided.")
            mask_path = random.choice(self.mask_pool)
            mask = self.load_mask(mask_path)
            if mask is None:
                raise RuntimeError(f"Cannot load mask: {mask_path}")
            # 确保掩码尺寸与原图一致（稍后统一 resize，此处先调整到原图尺寸）
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            # 验证：从 jsonl 读取固定掩码
            mask_path = rec.get("mask", None)
            if mask_path is None:
                raise ValueError("Validation record missing 'mask' field.")
            mask = self.load_mask(mask_path)
            if mask is None:
                raise RuntimeError(f"Cannot load mask: {mask_path}")
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # 生成损坏图像（将 mask 区域置零）
        damaged_image = image.copy()
        damaged_image[mask > 127] = 0

        # ---------- 加载结构先验 ----------
        if os.path.exists(skeleton_npz_path):
            S, Tx, Ty = load_skeleton_npz(skeleton_npz_path)
        else:
            S = np.zeros((H, W), dtype=np.float32)
            Tx = np.zeros((H, W), dtype=np.float32)
            Ty = np.zeros((H, W), dtype=np.float32)

        # ---------- 统一缩放和裁剪 ----------
        h, w = image.shape[:2]
        if w > h:
            scale = self.resolution / h
        else:
            scale = self.resolution / w
        w_new = int(np.ceil(w * scale))
        h_new = int(np.ceil(h * scale))

        # resize
        image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
        damaged_image = cv2.resize(damaged_image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        S = cv2.resize(S, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        Tx = cv2.resize(Tx, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        Ty = cv2.resize(Ty, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        # 随机/中心裁剪
        if self.is_train:
            x = random.randint(0, w_new - self.resolution) if w_new > self.resolution else 0
            y = random.randint(0, h_new - self.resolution) if h_new > self.resolution else 0
        else:
            x = (w_new - self.resolution) // 2 if w_new > self.resolution else 0
            y = (h_new - self.resolution) // 2 if h_new > self.resolution else 0

        image = image[y:y+self.resolution, x:x+self.resolution, :]
        damaged_image = damaged_image[y:y+self.resolution, x:x+self.resolution, :]
        mask = mask[y:y+self.resolution, x:x+self.resolution]
        S = S[y:y+self.resolution, x:x+self.resolution]
        Tx = Tx[y:y+self.resolution, x:x+self.resolution]
        Ty = Ty[y:y+self.resolution, x:x+self.resolution]

        # 归一化到 [-1, 1]
        # ---------- 数据增强（仅训练模式） ----------
        # 归一化到 [-1, 1]
        # ---------- 数据增强（仅训练模式） ----------
        if self.is_train:
            # 水平翻转 (50% 概率)
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                damaged_image = np.fliplr(damaged_image).copy()
                mask = np.fliplr(mask).copy()
                S = np.fliplr(S).copy()
                Tx = np.fliplr(Tx).copy()
                Ty = np.fliplr(Ty).copy()
                Tx = -Tx   # 水平镜像时方向向量的 x 分量取反
            # ---------- 随机旋转（仅训练模式） ----------
            if random.random() < 0.3:   # 30% 概率旋转
                angle = random.uniform(-3.0, 3.0)  # 小角度 ±3°
                if abs(angle) > 1e-3:
                    image, damaged_image, mask, S, Tx, Ty = rotate_image_and_mask(
                        image, damaged_image, mask, S, Tx, Ty, angle
                    )

        # 颜色抖动（仅训练模式）
        if self.is_train:
            # 亮度、对比度、饱和度调整因子
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)

            # 将图像转换为 PIL 方便增强（也可以直接用 numpy 操作）
            pil_img = Image.fromarray(image)
            pil_damaged = Image.fromarray(damaged_image)

            # 亮度
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Brightness(pil_damaged)
            pil_damaged = enhancer.enhance(brightness)

            # 对比度
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Contrast(pil_damaged)
            pil_damaged = enhancer.enhance(contrast)

            # 饱和度
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(saturation)
            enhancer = ImageEnhance.Color(pil_damaged)
            pil_damaged = enhancer.enhance(saturation)

            # 转回 numpy
            image = np.array(pil_img)
            damaged_image = np.array(pil_damaged)
        # 归一化到 [-1, 1]
        image = (image.astype(np.float32) / 127.5) - 1.0
        damaged_image = (damaged_image.astype(np.float32) / 127.5) - 1.0

        # 转为 tensor
        pv = torch.tensor(image).permute(2, 0, 1).float()
        cpv = torch.tensor(damaged_image).permute(2, 0, 1).float()
        m = torch.tensor(mask.astype(np.float32) / 255.0).unsqueeze(0).float()

        input_id = self.tokenize_caption(rec)  
        # 随机丢弃结构先验 (10% 概率)
        if self.is_train and random.random() < 0.1:
            S = np.zeros_like(S)
            Tx = np.zeros_like(Tx)
            Ty = np.zeros_like(Ty)

        S_tensor = torch.from_numpy(S).unsqueeze(0).float()
        Tx_tensor = torch.from_numpy(Tx).unsqueeze(0).float()
        Ty_tensor = torch.from_numpy(Ty).unsqueeze(0).float()

        return {
            "pixel_values": pv,
            "conditioning_pixel_values": cpv,
            "masks": m,
            "input_ids": input_id,
            "structure_S": S_tensor,
            "structure_Tx": Tx_tensor,
            "structure_Ty": Ty_tensor,
            "image_path": image_path,
            "mask_path": mask_path,   # 记录使用的掩码（可选）
            "damaged_path": "",       # 不再使用
            "caption_str": rec.get('prompt', ''),
            "skeleton_npz": skeleton_npz_path,  # 补充字段，用于验证集生成
        }

def collate_local(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch]).contiguous().float()
    conditioning_pixel_values = torch.stack([item["conditioning_pixel_values"] for item in batch]).contiguous().float()
    masks = torch.stack([item["masks"] for item in batch]).contiguous().float()
    input_ids = torch.stack([item["input_ids"] for item in batch]).contiguous().long()
    structure_S = torch.stack([item["structure_S"] for item in batch]).contiguous().float()
    structure_Tx = torch.stack([item["structure_Tx"] for item in batch]).contiguous().float()
    structure_Ty = torch.stack([item["structure_Ty"] for item in batch]).contiguous().float()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "masks": masks,
        "input_ids": input_ids,
        "structure_S": structure_S,
        "structure_Tx": structure_Tx,
        "structure_Ty": structure_Ty,
    }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.condif_model_name_or_path:
        logger.info("Loading existing condif weights")
        condif = CondifModel.from_pretrained(args.condif_model_name_or_path)
    else:
        logger.info("Initializing condif weights from unet")
        condif = CondifModel.from_unet(
            unet,
            conditioning_channels=8   # 8通道：4+1+1+2
        )

    # 创建训练集
    train_dataset = LocalCondifDataset(
        jsonl_path=args.train_data_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
        random_mask=args.random_mask,                 # 保留原参数，但内部逻辑已改为从文件加载
        proportion_empty_prompts=0.0,                 # 强制为空提示，与 ZITS 一致
        is_train=True,
        mask_file_list=args.mask_file_list             # 传入训练掩码列表
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_local,
        pin_memory=True,
        prefetch_factor=4,  # 🔥 加上这一行
        persistent_workers=True # 🔥 建议加上这一行，配合 num_workers 使用
    )
    train_dataloader_len = len(train_dataloader)
    train_dataset_len = len(train_dataset)

    # 创建验证集（如果提供了验证集路径）
    val_dataloader = None
    if args.val_data_dir is not None:
        val_dataset = LocalCondifDataset(
            jsonl_path=args.val_data_dir,
            tokenizer=tokenizer,
            resolution=args.resolution,
            random_mask=False,                          # 验证集不使用随机mask
            proportion_empty_prompts=0.0,               # 强制为空提示
            is_train=False,                              # 验证模式
            mask_file_list=None                          # 验证集不使用掩码池
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_local,
            pin_memory=True
        )
        logger.info(f"Validation set loaded: {len(val_dataset)} samples")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 定义缺失的low_precision_error_string变量
    low_precision_error_string = (
        "ConDiF loaded in low precision, which is not supported for training. "
        "Please load the model in float32."
    )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "condif"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                condif_dir = os.path.join(input_dir, "condif")
                if os.path.isdir(condif_dir):
                    load_model = CondifModel.from_pretrained(input_dir, subfolder="condif")
                else:
                    load_model = CondifModel.from_pretrained(input_dir)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    condif.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            condif.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        condif.enable_gradient_checkpointing()

    if unwrap_model(condif).dtype != torch.float32:
        raise ValueError(
            f"ConDiF loaded as datatype {unwrap_model(condif).dtype}. {low_precision_error_string}"
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = condif.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything
    # Prepare everything
    if val_dataloader is not None:
        condif, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            condif, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    else:
        condif, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            condif, optimizer, train_dataloader, lr_scheduler
        )

    # ✅ EMA 初始化（必须在 prepare 之后，使用 unwrapped model
    # 获取未包装的模型参数（accelerator.unwrap_model 返回原始模型）
    ema_condif = EMAModel(
        accelerator.unwrap_model(condif).parameters(), 
        decay=0.999
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # 初始化 SSIM 损失模块
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompt", None)
        tracker_config.pop("validation_image", None)
        tracker_config.pop("validation_mask", None)
        tracker_config.pop("random_mask", None)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

        # ========== 设置 wandb 使用 epoch 作为 x 轴（仅主进程需要） ==========
        if args.report_to == "wandb":
            import wandb
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")

    # ===== 将 total_batch_size 移出条件块，确保所有进程都能访问 =====
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num batches each epoch = {train_dataloader_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # 处理检查点恢复
    if args.resume_from_checkpoint:
        # 如果路径是绝对路径或包含 '/'，直接使用
        if os.path.exists(args.resume_from_checkpoint):
            path = args.resume_from_checkpoint
            checkpoint_dir = path
        else:
            # 否则按原有逻辑（从 output_dir 下找）
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
                checkpoint_dir = os.path.join(args.output_dir, path)
            else:
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                checkpoint_dir = os.path.join(args.output_dir, path) if path else None

        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            # 处理不存在的情况
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_dir}")
            accelerator.load_state(checkpoint_dir, map_location="cpu")
            global_step = int(os.path.basename(checkpoint_dir).split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # 初始化最佳验证损失
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)

    # 验证函数（内部定义以捕获变量）
    @torch.no_grad()
    def compute_validation_loss(num_batches=20):
        """
        在验证集上计算 loss，最多只跑 num_batches 个 batch。
        同时将相关模型设置为 eval 模式，计算完后恢复 train 模式。
        """
        accelerator.wait_for_everyone()
        if val_dataloader is None:
            return None

        # 保存原始模式，并设置为 eval
        unet.eval()
        vae.eval()
        text_encoder.eval()
        condif.eval()

        dtype = next(unet.parameters()).dtype
        device = accelerator.device
        total_loss = 0.0
        num_batches_done = 0

        print(f"[验证] 开始，最多验证 {num_batches} 个 batch，设备 {device}")
        start_time = time.time()
        total_batches = min(len(val_dataloader), num_batches)

        for i, batch in enumerate(val_dataloader):
            if i >= num_batches:
                break

            # 每 10 个 batch 或最后一个打印进度
            if i % 10 == 0 or i == total_batches - 1:
                elapsed = time.time() - start_time
                print(f"  处理 batch {i+1}/{total_batches}，已耗时 {elapsed:.2f}s")

            # 将 batch 中的张量移到设备并转 dtype
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            conditioning_pixel_values = batch["conditioning_pixel_values"].to(device=device, dtype=dtype)
            masks = batch["masks"].to(device=device, dtype=dtype)
            input_ids = batch["input_ids"].to(device=device)
            structure_S = batch["structure_S"].to(device=device, dtype=dtype)
            structure_Tx = batch["structure_Tx"].to(device=device, dtype=dtype)
            structure_Ty = batch["structure_Ty"].to(device=device, dtype=dtype)

            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            conditioning_latents = vae.encode(conditioning_pixel_values).latent_dist.sample() * vae.config.scaling_factor

            masks = F.interpolate(masks, size=latents.shape[-2:])

            S_latent = F.interpolate(structure_S, size=latents.shape[-2:], mode='bilinear', align_corners=False)
            Tx_latent = F.interpolate(structure_Tx, size=latents.shape[-2:], mode='bilinear', align_corners=False)
            Ty_latent = F.interpolate(structure_Ty, size=latents.shape[-2:], mode='bilinear', align_corners=False)
            direction_latent = torch.cat([Tx_latent, Ty_latent], dim=1)

            conditioning_latents = torch.cat([conditioning_latents, masks, S_latent, direction_latent], 1)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]


            down_block_res_samples, mid_block_res_sample, up_block_res_samples = condif(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                condif_cond=conditioning_latents,
                return_dict=False,
                cross_attention_kwargs={"direction_map": None},  # 🔥 必加！强制参数参与计算
            )
            # 确保传递给 unet 的中间样本也是正确的 dtype 和设备
            down_block_res_samples = [s.to(device=device, dtype=dtype) for s in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample.to(device=device, dtype=dtype)
            up_block_res_samples = [s.to(device=device, dtype=dtype) for s in up_block_res_samples]

            model_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                down_block_add_samples=down_block_res_samples,
                mid_block_add_sample=mid_block_res_sample,
                up_block_add_samples=up_block_res_samples,
                return_dict=False
            )[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # 原有的主损失计算
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            total_loss += loss.item()
            num_batches_done += 1

        # 恢复训练模式
        unet.train()
        vae.train()
        text_encoder.train()
        condif.train()

        avg_loss = total_loss / num_batches_done if num_batches_done > 0 else None
        total_time = time.time() - start_time
        print(f"[验证] 完成，实际验证 batch 数 = {num_batches_done}，总耗时 {total_time:.2f}s，平均 loss = {avg_loss:.6f}")
        return avg_loss

    
    @torch.no_grad()
    def log_validation_from_dataset(vae, text_encoder, tokenizer, unet, condif,
                                    val_dataloader, accelerator, weight_dtype, step, num_samples=4):
        """从验证集随机选取样本进行生成验证，并将结果记录到 wandb。"""
        if not accelerator.is_main_process:
            return
        condif = accelerator.unwrap_model(condif)
        condif.eval()

        # Build validation pipeline
        pipeline = build_inpainting_pipeline(
            args.pretrained_model_name_or_path,
            condif,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            safety_checker=None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

        # 随机选取样本
        dataset = val_dataloader.dataset
        total = len(dataset)
        indices = random.sample(range(total), min(num_samples, total))
        samples = [dataset[i] for i in indices]

        image_logs = []
        for i, sample in enumerate(samples):
            # 从样本中获取原始图像、mask、npz 路径（字段名与 __getitem__ 返回一致）
            image_path = sample.get("image_path")          # 原始图像路径
            mask_path = sample.get("mask_path")            # mask 路径
            skeleton_npz_path = sample.get("skeleton_npz") # 字段名是 "skeleton_npz"

            if not all([image_path, mask_path, skeleton_npz_path]):
                print(f"警告：样本 {i} 缺少必要路径，跳过")
                continue

            if not os.path.exists(image_path) or not os.path.exists(mask_path) or not os.path.exists(skeleton_npz_path):
                print(f"警告：样本 {i} 的图像文件不存在，跳过")
                continue

            # 加载原始图像和 mask
            original_img = Image.open(image_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            if original_img.size != mask_img.size:
                mask_img = mask_img.resize(original_img.size, Image.NEAREST)

            # 动态生成损坏图像（白色 mask 区域置黑）
            damaged_np = np.array(original_img)
            mask_np = np.array(mask_img) > 127
            damaged_np[mask_np] = 0
            damaged_img = Image.fromarray(damaged_np)

            # 提示词为空（与训练一致）
            caption = ""

            # 验证时不使用 CFG，guidance_scale 设为 1.0，因此 skeleton_paths 只需一个路径
            skeleton_paths = [skeleton_npz_path]

            with torch.autocast("cuda"):
                image = pipeline(
                    caption, 
                    damaged_img, 
                    mask_img,
                    skeleton_npz_paths=skeleton_paths,   # 传递结构先验
                    num_inference_steps=20,
                    generator=generator,
                    guidance_scale=1.0                    # 显式设置 guidance_scale=1.0 避免 CFG
                ).images[0]

            image_logs.append(wandb.Image(image, caption=caption))

        # 记录到 wandb
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log({"validation_samples": image_logs}, step=step)
            else:
                logger.warning(f"Image logging not implemented for {tracker.name}")

        condif.train()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    
    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        condif.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(condif):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                conditioning_latents = vae.encode(batch["conditioning_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                conditioning_latents = conditioning_latents * vae.config.scaling_factor

                masks = torch.nn.functional.interpolate(batch["masks"], size=latents.shape[-2:])

                S = batch["structure_S"]
                Tx = batch["structure_Tx"]
                Ty = batch["structure_Ty"]

                S_latent = torch.nn.functional.interpolate(S, size=latents.shape[-2:], mode='bilinear', align_corners=False)
                Tx_latent = torch.nn.functional.interpolate(Tx, size=latents.shape[-2:], mode='bilinear', align_corners=False)
                Ty_latent = torch.nn.functional.interpolate(Ty, size=latents.shape[-2:], mode='bilinear', align_corners=False)

                direction_latent = torch.cat([Tx_latent, Ty_latent], dim=1)

                conditioning_latents = torch.cat([
                    conditioning_latents,
                    masks,
                    S_latent,
                    direction_latent
                ], 1)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                down_block_res_samples, mid_block_res_sample, up_block_res_samples = condif(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    condif_cond=conditioning_latents,
                    return_dict=False,
                    cross_attention_kwargs={"direction_map": None},  # 🔥 必加！强制参数参与计算
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_add_samples=[s.to(dtype=weight_dtype) for s in down_block_res_samples],
                    mid_block_add_sample=mid_block_res_sample.to(dtype=weight_dtype),
                    up_block_add_samples=[s.to(dtype=weight_dtype) for s in up_block_res_samples],
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                # 基础 MSE 损失
                # 基础 MSE 损失
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 每 5 步计算一次 SSIM 损失
                # 每 20 步计算一次 SSIM 损失
#                 if global_step % 20 == 0:
#                     alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1,1,1,1)
#                     pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * model_pred) / alpha_prod_t.sqrt()
#                     pred_x0 = pred_x0.to(weight_dtype)
#                     latents_dtype = latents.to(weight_dtype)
#                     with torch.no_grad():
#                         pred_img = vae.decode(pred_x0 / vae.config.scaling_factor).sample
#                         target_img = vae.decode(latents_dtype / vae.config.scaling_factor).sample
#                         pred_img = (pred_img / 2 + 0.5).clamp(0, 1)
#                         target_img = (target_img / 2 + 0.5).clamp(0, 1)
#                     ssim_val = ssim_fn(pred_img, target_img)
#                     ssim_loss = 1 - ssim_val
#                     loss = loss + 0.1 * ssim_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = condif.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                ema_condif.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # 定期保存检查点
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # 存储原始权重，然后复制 EMA 权重
                        ema_condif.store(accelerator.unwrap_model(condif).parameters())
                        ema_condif.copy_to(accelerator.unwrap_model(condif).parameters())
                        accelerator.save_state(save_path)
                        # 恢复原始权重
                        ema_condif.restore(accelerator.unwrap_model(condif).parameters())
                        logger.info(f"Saved state to {save_path}")

                    # ---------- 验证逻辑（包含 EMA 权重切换） ----------
                    if args.val_data_dir is not None and global_step % args.validation_steps == 0:
                        # 验证前将 EMA 权重复制到模型
                        ema_condif.store(accelerator.unwrap_model(condif).parameters())
                        ema_condif.copy_to(accelerator.unwrap_model(condif).parameters())
                        val_loss = compute_validation_loss()
                        if val_loss is not None:
                            accelerator.log({"val_loss": val_loss}, step=global_step)
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                # 注意：此时 condif 已经是 EMA 权重，直接保存即可
                                unwrapped_condif = accelerator.unwrap_model(condif)
                                unwrapped_condif.save_pretrained(best_model_path)
                                logger.info(f"New best model saved with val_loss {val_loss:.4f}")

#                         log_validation_from_dataset(
#                             vae, text_encoder, tokenizer, unet, condif,
#                             val_dataloader, accelerator, weight_dtype, global_step, num_samples=4
#                         )
                        
                        # 验证后恢复原始权重
                        ema_condif.restore(accelerator.unwrap_model(condif).parameters())

                    # ---------- 原有图像验证逻辑 ----------
#                     if args.validation_prompt is not None and global_step % args.validation_steps == 0:
#                         image_logs = log_validation(
#                             vae,
#                             text_encoder,
#                             tokenizer,
#                             unet,
#                             condif,
#                             args,
#                             accelerator,
#                             weight_dtype,
#                             global_step,
#                         )

            # ---------- 日志记录（在 sync_gradients 外面，但在 step 循环里面） ----------
            current_epoch = epoch + (step + 1) / train_dataloader_len
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": current_epoch}
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # 训练结束，保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 使用 EMA 权重保存最终模型
        ema_condif.copy_to(accelerator.unwrap_model(condif).parameters())
        unwrapped_condif = accelerator.unwrap_model(condif)
        unwrapped_condif.save_pretrained(args.output_dir)
        # 可选：恢复原始权重（若之后还有操作）
        ema_condif.restore(accelerator.unwrap_model(condif).parameters())
        # 最终验证（可选）
#         if args.validation_prompt is not None:
#             image_logs = log_validation(
#                 vae=vae,
#                 text_encoder=text_encoder,
#                 tokenizer=tokenizer,
#                 unet=unet,
#                 condif=None,
#                 args=args,
#                 accelerator=accelerator,
#                 weight_dtype=weight_dtype,
#                 step=global_step,
#                 is_final_validation=True,
#             )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)