"""CORE: Contextual Safety Reasoning and Enforcement.

Reproduction of "Contextual Safety Reasoning and Grounding for Open-World
Robots" (Ravichandran et al., arXiv:2602.19983).

Three modules:
  - reasoning:  VLM-based contextual safety constraint inference
  - grounding:  segmentation -> spatial operators -> costmap -> barrier h(x)
  - control:    CBF-QP safety filter over control-affine dynamics
"""

__version__ = "0.1.0"
