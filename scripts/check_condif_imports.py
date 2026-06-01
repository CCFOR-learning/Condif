"""Quick import check before running evaluation."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

print("diffusers path:", os.path.join(project_root, "src"))

try:
    from diffusers import CondifModel, StableDiffusionCondifPipeline, DPMSolverMultistepScheduler
    print("OK: top-level import")
except ImportError as exc:
    print("Top-level import failed:", exc)
    from diffusers.models.condif import CondifModel
    from diffusers.pipelines.condif import StableDiffusionCondifPipeline
    print("OK: submodule import")

print("CondifModel:", CondifModel)
print("StableDiffusionCondifPipeline:", StableDiffusionCondifPipeline)
print("skeleton_npz_paths:", "skeleton_npz_paths" in StableDiffusionCondifPipeline.__call__.__code__.co_varnames)
