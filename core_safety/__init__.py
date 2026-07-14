"""Contextual safety reasoning and enforcement framework.

Three modules:
  - reasoning:  VLM-based contextual safety constraint inference
  - grounding:  segmentation -> spatial operators -> costmap -> barrier h(x)
  - control:    CBF-QP safety filter over control-affine dynamics
"""

__version__ = "0.1.0"
