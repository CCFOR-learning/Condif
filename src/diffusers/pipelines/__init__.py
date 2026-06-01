from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_transformers_available

_import_structure = {
    "pipeline_utils": ["DiffusionPipeline", "StableDiffusionMixin", "ImagePipelineOutput"],
    "condif": [],
}

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["condif"].extend(["StableDiffusionCondifPipeline"])

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .condif import StableDiffusionCondifPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
