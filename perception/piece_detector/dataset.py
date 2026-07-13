"""ChessReD piece-box dataset for GroundingDINO fine-tuning.

GroundingDINO is phrase-grounded: a single text prompt lists every class as a
period-separated phrase ("white pawn. white rook. ... black king."), and
`class_labels` for each box is that phrase's *position* in the prompt (0-11),
not an arbitrary category id -- transformers' `build_label_maps` derives the
mapping by splitting `input_ids` on the delimiter token, in prompt order. This
happens to line up directly with ChessReD's category_id (0-11 in exactly this
order in annotations.json, with "empty" as 12 excluded since it's never used
as a box label), but that alignment is what CATEGORY_NAMES encodes, not an
assumption baked elsewhere.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).parent / "data"

CATEGORY_NAMES = [
    "white pawn", "white rook", "white knight", "white bishop", "white queen", "white king",
    "black pawn", "black rook", "black knight", "black bishop", "black queen", "black king",
]  # fmt: skip
TEXT_PROMPT = ". ".join(CATEGORY_NAMES) + "."


class ChessRedDetectionDataset(Dataset):
    """One sample = one image + its piece boxes, in xyxy pixel coords."""

    def __init__(self, split_json: Path):
        with open(split_json) as f:
            data = json.load(f)
        self.images = data["images"]
        anns_by_image: dict[int, list] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        self.anns_by_image = anns_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        im = self.images[idx]
        image = Image.open(DATA_DIR / im["path"]).convert("RGB")

        boxes_xyxy = []
        class_labels = []
        for ann in self.anns_by_image.get(im["id"], []):
            x, y, w, h = ann["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            class_labels.append(ann["category_id"])

        return {
            "image": image,
            "boxes_xyxy": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "image_id": im["id"],
        }


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        text = [TEXT_PROMPT] * len(batch)

        encoding = processor(images=images, text=text, return_tensors="pt", padding=True)

        labels = []
        for b in batch:
            w, h = b["image"].size
            boxes = b["boxes_xyxy"].clone()
            # xyxy pixels -> cxcywh normalized, as GroundingDinoForObjectDetection expects
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
            bw = (boxes[:, 2] - boxes[:, 0]) / w
            bh = (boxes[:, 3] - boxes[:, 1]) / h
            norm_boxes = torch.stack([cx, cy, bw, bh], dim=1)
            labels.append({"class_labels": b["class_labels"], "boxes": norm_boxes})

        return encoding, labels, [b["image_id"] for b in batch]

    return collate_fn
