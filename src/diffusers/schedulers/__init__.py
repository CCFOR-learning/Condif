from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, is_scipy_available, is_torch_available

_import_structure = {}

if is_torch_available():
    _import_structure["scheduling_ddim"] = ["DDIMScheduler"]
    _import_structure["scheduling_ddpm"] = ["DDPMScheduler"]
    _import_structure["scheduling_dpmsolver_multistep"] = ["DPMSolverMultistepScheduler"]
    _import_structure["scheduling_edm_dpmsolver_multistep"] = ["EDMDPMSolverMultistepScheduler"]
    _import_structure["scheduling_euler_ancestral_discrete"] = ["EulerAncestralDiscreteScheduler"]
    _import_structure["scheduling_euler_discrete"] = ["EulerDiscreteScheduler"]
    _import_structure["scheduling_heun_discrete"] = ["HeunDiscreteScheduler"]
    _import_structure["scheduling_pndm"] = ["PNDMScheduler"]
    _import_structure["scheduling_unipc_multistep"] = ["UniPCMultistepScheduler"]
    _import_structure["scheduling_utils"] = ["KarrasDiffusionSchedulers", "SchedulerMixin"]
    if is_scipy_available():
        _import_structure["scheduling_lms_discrete"] = ["LMSDiscreteScheduler"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .scheduling_ddim import DDIMScheduler
        from .scheduling_ddpm import DDPMScheduler
        from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        from .scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
        from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        from .scheduling_euler_discrete import EulerDiscreteScheduler
        from .scheduling_heun_discrete import HeunDiscreteScheduler
        from .scheduling_pndm import PNDMScheduler
        from .scheduling_unipc_multistep import UniPCMultistepScheduler
        from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

        if is_scipy_available():
            from .scheduling_lms_discrete import LMSDiscreteScheduler
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
