__version__ = "0.27.0.dev0"

from typing import TYPE_CHECKING

from .utils import DIFFUSERS_SLOW_IMPORT, OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_transformers_available

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "image_processor": ["VaeImageProcessor"],
    "optimization": [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_torch_available",
        "is_transformers_available",
        "logging",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["models"] = [
        "AutoencoderKL",
        "CondifModel",
        "CondifOutput",
        "ImageProjection",
        "ModelMixin",
        "UNet2DConditionModel",
    ]
    _import_structure["schedulers"] = [
        "DDIMScheduler",
        "DDPMScheduler",
        "DPMSolverMultistepScheduler",
        "EDMDPMSolverMultistepScheduler",
        "EulerAncestralDiscreteScheduler",
        "EulerDiscreteScheduler",
        "HeunDiscreteScheduler",
        "PNDMScheduler",
        "UniPCMultistepScheduler",
        "KarrasDiffusionSchedulers",
        "SchedulerMixin",
    ]

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["pipelines"] = ["StableDiffusionCondifPipeline", "DiffusionPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin
    from .image_processor import VaeImageProcessor
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .utils import OptionalDependencyNotAvailable, is_torch_available, is_transformers_available, logging

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .models import AutoencoderKL, CondifModel, CondifOutput, ImageProjection, ModelMixin, UNet2DConditionModel
        from .schedulers import (
            DDIMScheduler,
            DDPMScheduler,
            DPMSolverMultistepScheduler,
            EDMDPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            PNDMScheduler,
            SchedulerMixin,
            UniPCMultistepScheduler,
        )
        from .schedulers.scheduling_utils import KarrasDiffusionSchedulers

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .pipelines import DiffusionPipeline, StableDiffusionCondifPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
