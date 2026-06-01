from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, is_torch_available

_import_structure = {}

if is_torch_available():
    _import_structure["condif"] = ["CondifModel", "CondifOutput"]
    _import_structure["autoencoders"] = ["AutoencoderKL"]
    _import_structure["embeddings"] = ["ImageProjection"]
    _import_structure["modeling_utils"] = ["ModelMixin"]
    _import_structure["unets"] = ["UNet2DConditionModel"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .autoencoders import AutoencoderKL
        from .condif import CondifModel, CondifOutput
        from .embeddings import ImageProjection
        from .modeling_utils import ModelMixin
        from .unets import UNet2DConditionModel
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
