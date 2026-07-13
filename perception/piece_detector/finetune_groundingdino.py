"""Fine-tune GroundingDINO (IDEA-Research/grounding-dino-tiny, via transformers)
on the ChessReD chessred2k piece-box subset.

Phase 3 note: SAM2 is used downstream of this detector's boxes as a frozen,
box-prompted segmenter, not separately fine-tuned -- ChessReD has no
segmentation-mask ground truth to fine-tune SAM2 against (see
perception/piece_detector/segment.py and CLAUDE.md).

Runs on a 6GB GPU: batch_size=1 (a batch of 2 already peaks at ~5.4GB and
risks OOM on image-heavy batches) with gradient accumulation for a larger
effective batch, fp16 autocast, and images downsized to 480px shortest edge
(deformable attention memory scales with feature-map size, and 800px -- the
model's default -- OOMs a single image alone on this GPU).

Run: python perception/piece_detector/finetune_groundingdino.py [--epochs N]
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, logging as hf_logging

from dataset import ChessRedDetectionDataset, DATA_DIR, make_collate_fn

hf_logging.set_verbosity_error()

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IMAGE_SIZE = {"shortest_edge": 480, "longest_edge": 640}
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def move_to_device(encoding, labels, device):
    encoding = {k: v.to(device) for k, v in encoding.items()}
    labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
    return encoding, labels


def build_balanced_sampler(train_json: Path) -> WeightedRandomSampler:
    """Per-image weight = 1/frequency of its rarest category, so images
    containing a queen or knight (the categories that end up confused with
    their prompt-neighbor -- see evaluate.py) get sampled more often than
    their ~2.7-4.6% share of boxes would otherwise give them.
    """
    with open(train_json) as f:
        data = json.load(f)
    freq = Counter(a["category_id"] for a in data["annotations"])
    cats_by_image: dict[int, list[int]] = {}
    for a in data["annotations"]:
        cats_by_image.setdefault(a["image_id"], []).append(a["category_id"])

    weights = []
    for im in data["images"]:
        cats = cats_by_image.get(im["id"], [])
        weight = max((1.0 / freq[c] for c in cats), default=1.0)
        weights.append(weight)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--limit-steps", type=int, default=None, help="for smoke-testing timing")
    parser.add_argument(
        "--resume", type=Path, default=None, help="checkpoint dir to continue training from, e.g. checkpoints/final"
    )
    parser.add_argument(
        "--epoch-offset", type=int, default=0, help="epoch number to start counting/naming checkpoints from"
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="oversample images containing rare categories (see build_balanced_sampler)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, size=IMAGE_SIZE)
    model_source = args.resume if args.resume is not None else MODEL_ID
    print(f"loading model weights from: {model_source}")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_source).to(device)
    model.train()

    train_ds = ChessRedDetectionDataset(DATA_DIR / "train.json")
    collate_fn = make_collate_fn(processor)
    if args.balanced_sampling:
        sampler = build_balanced_sampler(DATA_DIR / "train.json")
        loader = DataLoader(train_ds, batch_size=1, sampler=sampler, collate_fn=collate_fn, num_workers=2)
    else:
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    step = 0
    t0 = time.time()
    running_loss = 0.0
    for epoch in range(args.epoch_offset, args.epoch_offset + args.epochs):
        for encoding, labels, _image_ids in loader:
            encoding, labels = move_to_device(encoding, labels, device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                outputs = model(**encoding, labels=labels)
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"epoch {epoch} step {step}: loss={running_loss / args.log_every:.2f} "
                    f"({elapsed / step:.2f}s/step, {elapsed:.0f}s elapsed)"
                )
                running_loss = 0.0

            if args.limit_steps is not None and step >= args.limit_steps:
                print(f"stopping early at step {step} (--limit-steps)")
                return

        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}"
        model.save_pretrained(ckpt_path)
        print(f"epoch {epoch} done, checkpoint saved to {ckpt_path}")

    final_path = CHECKPOINT_DIR / "final"
    model.save_pretrained(final_path)
    print(f"training done, final checkpoint at {final_path}")


if __name__ == "__main__":
    main()
