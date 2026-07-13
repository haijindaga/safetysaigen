"""SAM3 open-vocabulary segmenter (paper's choice; Ubuntu/GPU only).

Uses HuggingFace transformers' SAM3 (Promptable Concept Segmentation):
text prompt per semantic class -> union of instance masks for that class.

Install (Ubuntu, inside the project venv):
    pip install "transformers>=5.0" torch --index-url https://download.pytorch.org/whl/cu121
    # weights auto-download from facebook/sam3 on first use (~3.4 GB)

Notes for the RTX 4060 (8 GB):
  - Run SAM3 in fp16 (`dtype="float16"`), ~2x less VRAM.
  - The VLM lives in Ollama's memory space (partially CPU-offloaded), so
    SAM3 + VLM coexist; if VRAM is tight, pass device="cpu" (slower).
  - `precompute_vision=True` embeds the image once and reuses it across
    class prompts (the per-frame class list is usually 3-6 classes).
"""
from __future__ import annotations

import numpy as np

from .segmentation import Segmenter


class SAM3Segmenter(Segmenter):
    def __init__(self, model_id: str = "facebook/sam3",
                 device: str | None = None,
                 dtype: str = "float16",
                 threshold: float = 0.5,
                 mask_threshold: float = 0.5):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.model = Sam3Model.from_pretrained(
            model_id, dtype=torch_dtype).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.threshold = threshold
        self.mask_threshold = mask_threshold

    def segment(self, rgb: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
        from PIL import Image
        torch = self.torch
        image = Image.fromarray(rgb).convert("RGB")
        h, w = rgb.shape[:2]

        # Embed the image once, reuse across class prompts.
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs.pixel_values)

        out: dict[str, np.ndarray] = {}
        for cls in classes:
            prompt = cls.replace("_", " ")
            text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=img_inputs.get("original_sizes").tolist())[0]
            mask = np.zeros((h, w), dtype=bool)
            for m in results["masks"]:
                mask |= np.asarray(m.cpu().numpy(), dtype=bool)
            out[cls] = mask
        return out
