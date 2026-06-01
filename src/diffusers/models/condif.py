from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os

import torch
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from .embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
    get_mid_block,
    get_up_block,
    MidBlock2D
)
from .unets.unet_2d_condition import UNet2DConditionModel

logger = logging.get_logger(__name__)

LEGACY_CONDIF_CLASS_NAMES = ("BrushNetModel", "BrushnetModel")


def migrate_legacy_condif_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Rename legacy BrushNet checkpoint metadata to ConDiF for public release."""
    config = dict(config_dict)
    if config.get("_class_name") in LEGACY_CONDIF_CLASS_NAMES:
        config["_class_name"] = "CondifModel"
    if "brushnet_conditioning_channel_order" in config:
        if "condif_conditioning_channel_order" not in config:
            config["condif_conditioning_channel_order"] = config["brushnet_conditioning_channel_order"]
        del config["brushnet_conditioning_channel_order"]
    name_or_path = config.get("_name_or_path", "")
    if isinstance(name_or_path, str) and "brushnet" in name_or_path.lower():
        config["_name_or_path"] = name_or_path.replace("brushnet", "condif").replace("BrushNet", "Condif")
    return config


def migrate_legacy_condif_config_file(config_path: str, write_back: bool = True) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    migrated = migrate_legacy_condif_config(config)
    if write_back and migrated != config:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(migrated, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.info("Migrated legacy config: %s", config_path)
    return migrated


@dataclass
class CondifOutput(BaseOutput):
    up_block_res_samples: Tuple[torch.Tensor]
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


def zero_module(module):
    """Zero-initialize the parameters of a module."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class FiLMGenerator(nn.Module):
    def __init__(self, cond_channels, out_feat_channels):
        super().__init__()
        mid = max(cond_channels, 32)
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_feat_channels * 2, kernel_size=3, padding=1),
        )
        # 最后一层用小随机数初始化
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)
class CondifModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        condif_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        num_attention_heads = num_attention_heads or attention_head_dim

        if len(down_block_types) != len(up_block_types):
            raise ValueError("down/up block count mismatch")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("block_out_channels and down_block_types mismatch")
        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError("only_cross_attention length mismatch")
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError("num_attention_heads length mismatch")
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        self._conditioning_channels = conditioning_channels
        self.cond_mask_channel = 4
        self.cond_S_channel = 5
        self.cond_Tx_channel = 6
        self.cond_Ty_channel = 7

        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in_condition = nn.Conv2d(
            in_channels + conditioning_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding
        )

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError("encoder_hid_dim must be set when encoder_hid_dim_type is provided")
        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        else:
            self.encoder_hid_proj = None

        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("projection requires projection_class_embeddings_input_dim")
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            text_time_embedding_from_dim = encoder_hid_dim if encoder_hid_dim is not None else cross_attention_dim
            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim,
                image_embed_dim=cross_attention_dim,
                time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.add_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.condif_down_blocks = nn.ModuleList([])
        self.film_generators_down = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        output_channel = block_out_channels[0]
        condif_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        condif_block = zero_module(condif_block)
        self.condif_down_blocks.append(condif_block)
        self.film_generators_down.append(FiLMGenerator(self._conditioning_channels, output_channel))

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                condif_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                condif_block = zero_module(condif_block)
                self.condif_down_blocks.append(condif_block)
                self.film_generators_down.append(FiLMGenerator(self._conditioning_channels, output_channel))

            if not is_final_block:
                condif_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                condif_block = zero_module(condif_block)
                self.condif_down_blocks.append(condif_block)
                self.film_generators_down.append(FiLMGenerator(self._conditioning_channels, output_channel))

        mid_block_channel = block_out_channels[-1]
        condif_mid_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        condif_mid_block = zero_module(condif_mid_block)
        self.condif_mid_block = condif_mid_block
        self.film_generator_mid = FiLMGenerator(self._conditioning_channels, mid_block_channel)

        self.mid_block = get_mid_block(
            mid_block_type,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        self.up_blocks = nn.ModuleList([])
        self.condif_up_blocks = nn.ModuleList([])
        self.film_generators_up = nn.ModuleList([])

        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            add_upsample = not is_final_block
            if add_upsample:
                self.num_upsamplers += 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=reversed_num_attention_heads[i],
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

            for _ in range(layers_per_block + 1):
                condif_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                condif_block = zero_module(condif_block)
                self.condif_up_blocks.append(condif_block)
                self.film_generators_up.append(FiLMGenerator(self._conditioning_channels, output_channel))

            if not is_final_block:
                condif_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                condif_block = zero_module(condif_block)
                self.condif_up_blocks.append(condif_block)
                self.film_generators_up.append(FiLMGenerator(self._conditioning_channels, output_channel))

    @classmethod

    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        condif_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = 8,
    ):
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        # ========== 自定义块类型：只保留关键层的交叉注意力 ==========
        custom_down_block_types = [
            "DownBlock2D",            # 64x64
            "CrossAttnDownBlock2D",   # 32x32
            "DownBlock2D",            # 16x16
            "DownBlock2D",            # 8x8
        ]
        custom_mid_block_type = "UNetMidBlock2DCrossAttn"   # 中间块 8x8
        custom_up_block_types = [
            "UpBlock2D",              # 8->16
            "UpBlock2D",              # 16->32
            "CrossAttnUpBlock2D",     # 32->64
            "UpBlock2D",              # 64->128
        ]
        # =======================================================

        condif_model = cls(
            in_channels=unet.config.in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=custom_down_block_types,      # 使用自定义列表
            mid_block_type=custom_mid_block_type,          # 使用自定义中间块
            up_block_types=custom_up_block_types,          # 使用自定义上采样块
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            condif_conditioning_channel_order=condif_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            conv_in_condition_weight = torch.zeros_like(condif_model.conv_in_condition.weight)
            conv_in_condition_weight[:, :4, ...] = unet.conv_in.weight
            conv_in_condition_weight[:, 4:8, ...] = unet.conv_in.weight
            condif_model.conv_in_condition.weight = torch.nn.Parameter(conv_in_condition_weight)
            condif_model.conv_in_condition.bias = unet.conv_in.bias

            condif_model.time_proj.load_state_dict(unet.time_proj.state_dict())
            condif_model.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if condif_model.class_embedding:
                condif_model.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            condif_model.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)
            condif_model.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False)
            condif_model.up_blocks.load_state_dict(unet.up_blocks.state_dict(), strict=False)

        return condif_model
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError("Cannot set default processor due to mixed processor types.")
        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        sliceable_head_dims = []
        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)
        if slice_size == "auto":
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = [1] * num_sliceable_layers
        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size
        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError("slice_size length mismatch")

        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)
        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    @classmethod
    def load_config(cls, *args, **kwargs):
        config, unused_kwargs, commit_hash = super().load_config(*args, **kwargs)
        return migrate_legacy_condif_config(config), unused_kwargs, commit_hash

    def load_state_dict(self, state_dict, strict=True):
        remapped = {}
        for key, value in state_dict.items():
            remapped[key.replace("brushnet_", "condif_")] = value
        return super().load_state_dict(remapped, strict=strict)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        condif_cond: Optional[torch.FloatTensor] = None,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[CondifOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
        if condif_cond is None:
            raise ValueError("`condif_cond` must be provided.")
        cond = condif_cond

        channel_order = getattr(self.config, "condif_conditioning_channel_order", "rgb")
        if channel_order == "bgr":
            cond = torch.flip(cond, dims=[1])

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            dtype = torch.float32 if is_mps else torch.float64 if isinstance(timestep, float) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError("addition_embed_type 'text_time' requires 'text_embeds'")
                text_embeds = added_cond_kwargs["text_embeds"]
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError("addition_embed_type 'text_time' requires 'time_ids'")
                time_ids = added_cond_kwargs["time_ids"]
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)
        emb = emb + aug_emb if aug_emb is not None else emb

        # 输入拼接
        condif_cond_cat = torch.cat([sample, cond], dim=1)
        sample = self.conv_in_condition(condif_cond_cat)

        # 下采样
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # FiLM + zero-conv injection (down path)
        condif_down_block_res_samples = ()
        for d_res, condif_block, film_gen in zip(down_block_res_samples, self.condif_down_blocks, self.film_generators_down):
            x = condif_block(d_res)
            cond_resized = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
            gamma_beta = film_gen(cond_resized)
            gamma, beta = gamma_beta.chunk(2, dim=1)

            if cond_resized.shape[1] > self.cond_mask_channel:
                mask = cond_resized[:, self.cond_mask_channel:self.cond_mask_channel+1, :, :]
                if mask.shape[2:] != gamma.shape[2:]:
                    mask = F.interpolate(mask, size=gamma.shape[2:], mode='nearest')
                gamma = 1.0 + mask * gamma
                beta = mask * beta
            else:
                gamma = 1.0 + gamma
                beta = beta

            x = gamma * x + beta
            condif_down_block_res_samples += (x,)

        # 中间块
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample, emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

        # FiLM + 中间块
        condif_mid = self.condif_mid_block(sample)
        cond_resized_mid = F.interpolate(cond, size=condif_mid.shape[2:], mode='bilinear', align_corners=False)
        gamma_beta_mid = self.film_generator_mid(cond_resized_mid)
        gamma_mid, beta_mid = gamma_beta_mid.chunk(2, dim=1)
        if cond_resized_mid.shape[1] > self.cond_mask_channel:
            mask_mid = cond_resized_mid[:, self.cond_mask_channel:self.cond_mask_channel+1, :, :]
            gamma_mid = 1.0 + mask_mid * gamma_mid
            beta_mid = mask_mid * beta_mid
        else:
            gamma_mid = 1.0 + gamma_mid
            beta_mid = beta_mid
        condif_mid = gamma_mid * condif_mid + beta_mid

        # 上采样
        up_block_res_samples = ()
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            upsample_size = down_block_res_samples[-1].shape[2:] if not is_final_block else None

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    return_res_samples=True
                )
            else:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    return_res_samples=True
                )
            up_block_res_samples += up_res_samples

        # FiLM + zero-conv injection (up path)
        condif_up_block_res_samples = ()
        for u_res, condif_block, film_gen in zip(up_block_res_samples, self.condif_up_blocks, self.film_generators_up):
            x = condif_block(u_res)
            cond_resized = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
            gamma_beta = film_gen(cond_resized)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            if cond_resized.shape[1] > self.cond_mask_channel:
                mask = cond_resized[:, self.cond_mask_channel:self.cond_mask_channel+1, :, :]
                gamma = 1.0 + mask * gamma
                beta = mask * beta
            else:
                gamma = 1.0 + gamma
                beta = beta
            x = gamma * x + beta
            condif_up_block_res_samples += (x,)

        # 缩放
        if guess_mode and not self.config.global_pool_conditions:
            total_len = len(condif_down_block_res_samples) + 1 + len(condif_up_block_res_samples)
            scales = torch.logspace(-1, 0, total_len, device=sample.device) * conditioning_scale
            condif_down_block_res_samples = [
                s * scale for s, scale in zip(condif_down_block_res_samples, scales[:len(condif_down_block_res_samples)])
            ]
            condif_mid = condif_mid * scales[len(condif_down_block_res_samples)]
            condif_up_block_res_samples = [
                s * scale for s, scale in zip(condif_up_block_res_samples, scales[len(condif_down_block_res_samples)+1:])
            ]
        else:
            condif_down_block_res_samples = [s * conditioning_scale for s in condif_down_block_res_samples]
            condif_mid = condif_mid * conditioning_scale
            condif_up_block_res_samples = [s * conditioning_scale for s in condif_up_block_res_samples]

        if self.config.global_pool_conditions:
            condif_down_block_res_samples = [torch.mean(s, dim=(2,3), keepdim=True) for s in condif_down_block_res_samples]
            condif_mid = torch.mean(condif_mid, dim=(2,3), keepdim=True)
            condif_up_block_res_samples = [torch.mean(s, dim=(2,3), keepdim=True) for s in condif_up_block_res_samples]

        if not return_dict:
            return (condif_down_block_res_samples, condif_mid, condif_up_block_res_samples)

        return CondifOutput(
            down_block_res_samples=condif_down_block_res_samples,
            mid_block_res_sample=condif_mid,
            up_block_res_samples=condif_up_block_res_samples
        )