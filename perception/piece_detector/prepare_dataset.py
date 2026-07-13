"""Filter the full ChessReD annotations.json down to the 2078 images we have on
disk (the chessred2k subset -- see CLAUDE.md / project memory for why we're
using this subset instead of the full 24.6GB image set), and split into
train/val/test COCO-style JSON files using ChessReD's own chessred2k split.

annotations.json's `pieces` list covers all 10,800 images; `corners` and the
`splits.chessred2k` split are already scoped to exactly our 2078 images, which
is what confirms we have the complete intended subset (2078 == 1442+330+306).

Run: python perception/piece_detector/prepare_dataset.py
Writes: perception/piece_detector/data/{train,val,test}.json
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SRC_ANNOTATIONS = DATA_DIR / "annotations.json"


def main() -> None:
    with open(SRC_ANNOTATIONS) as f:
        src = json.load(f)

    images_by_id = {im["id"]: im for im in src["images"]}
    categories = src["categories"]

    pieces_by_image: dict[int, list] = {}
    for ann in src["annotations"]["pieces"]:
        pieces_by_image.setdefault(ann["image_id"], []).append(ann)

    split_def = src["splits"]["chessred2k"]

    total_written = 0
    for split_name in ("train", "val", "test"):
        image_ids = split_def[split_name]["image_ids"]
        images = []
        annotations = []
        missing_files = 0
        for image_id in image_ids:
            im = images_by_id[image_id]
            if not (DATA_DIR / im["path"]).exists():
                missing_files += 1
                continue
            images.append(im)
            annotations.extend(pieces_by_image.get(image_id, []))

        out = {"images": images, "annotations": annotations, "categories": categories}
        out_path = DATA_DIR / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f)

        print(
            f"{split_name}: {len(images)} images "
            f"({missing_files} missing on disk), {len(annotations)} piece boxes -> {out_path}"
        )
        total_written += len(images)

    print(f"\n{total_written} images total across train/val/test")


if __name__ == "__main__":
    main()
