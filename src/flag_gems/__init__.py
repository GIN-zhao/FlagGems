import torch

from . import testing  # noqa: F401
from .fused import *  # noqa: F403
from .ops import *  # noqa: F403

__version__ = "2.1"

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("addmm", addmm, "CUDA")

    lib.impl("bmm", bmm, "CUDA")

    lib.impl("mm", mm, "CUDA")
  

class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib


__all__ = [
    "enable",
    "use_gems",
]
