"""Box-prompted segmentation via a frozen, pretrained SAM2.

Per Phase 3 decision (see project memory / CLAUDE.md): ChessReD has no
segmentation-mask ground truth, so there's nothing to fine-tune SAM2 against.
It's used zero-shot here -- exactly the standard Grounded-SAM pattern, where
an open-vocabulary detector (our fine-tuned GroundingDINO) supplies boxes and
a promptable segmenter turns each box into a mask.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"


class BoxPromptedSegmenter:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
        self.model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def segment(self, image: Image.Image, boxes_xyxy: list[list[float]]) -> np.ndarray:
        """boxes_xyxy: list of [x0,y0,x1,y1] in pixel coords. Returns masks (N,H,W) bool."""
        if not boxes_xyxy:
            w, h = image.size
            return np.zeros((0, h, w), dtype=bool)

        inputs = self.processor(
            images=image,
            input_boxes=[boxes_xyxy],
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]
        # masks: (num_boxes, 1, H, W) -> (num_boxes, H, W)
        return masks.squeeze(1).cpu().numpy() > 0
