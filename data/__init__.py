"""Data layer: LS-DYNA loader protocol and IC builder.

Provides two back-ends that satisfy the ``LSDYNALoader`` protocol:
- ``AnalyticalLoader``  – synthesises data from the JWL+Sedov analytical
  pre-processing (no LS-DYNA files needed; default for v1).
- ``FileLoader``        – reads real LS-DYNA d3plot/CSV outputs (interface
  only; raises NotImplementedError until the user supplies data).
"""

from .lsdyna_loader import AnalyticalLoader, FileLoader, LSDYNALoader
from .ic_builder import ICBuilder

__all__ = ["AnalyticalLoader", "FileLoader", "LSDYNALoader", "ICBuilder"]
