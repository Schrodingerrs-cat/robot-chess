"""Evaluate a fine-tuned GroundingDINO checkpoint on the ChessReD test split,
reporting per-piece-type detection accuracy -- the Phase 3 deliverable.

Deliberately does NOT use processor.post_process_grounded_object_detection's
text-label decoding: that function extracts a label by finding the contiguous
run of tokens whose probability exceeds text_threshold and decoding all of
them together, and when the model's confidence for two adjacent categories in
the prompt ("white knight. white bishop.") both exceed threshold, it merges
them into one garbled string ("white knight white bishop") instead of picking
one. That's a real failure mode here (see below), not a rare edge case, so it
would silently misreport per-category accuracy as nearly zero for any
category the model is still learning to separate from its prompt-neighbor.

Instead: for each of GroundingDINO's 900 queries, take the mean sigmoid
probability over each category's own token span (found directly from
input_ids by splitting on "." token boundaries) and argmax over the 12
categories -- this reads the model's actual per-category confidence directly,
independent of the string decoder. Ground-truth boxes are matched to
predicted queries by IoU (>0.5) rather than by re-running NMS/thresholding,
since the question here is "does the model's own top query at this location
know which piece this is," not "would a full detection pipeline surface it."

Training history: after the initial 15 flat-sampled epochs, pawn/rook/king
were already accurate (>95%+) but knight and queen were confused with their
immediate prompt-neighbor (bishop and king respectively) close to 100% of the
time -- ruled out as a prompt-adjacency artifact via a same-checkpoint
prompt-reordering test (identical results either way), and ruled out as pure
class-imbalance for knight specifically (its box count is comparable to
bishop's, which converged fine). Two rounds of --balanced-sampling (8 epochs
each, 15 total additional epochs) resolved both: final checkpoint reaches
87.9% mean per-category accuracy, worst category (knight) at 50%+. See
project memory (project_phase3_knight_queen_finding) for the full diagnostic
trail if this regresses on a future retrain.

Run: python perception/piece_detector/evaluate.py [--checkpoint DIR]
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, logging as hf_logging
from transformers.image_transforms import center_to_corners_format

from dataset import CATEGORY_NAMES, TEXT_PROMPT, DATA_DIR

hf_logging.set_verbosity_error()

BASE_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IMAGE_SIZE = {"shortest_edge": 480, "longest_edge": 640}
IOU_MATCH_THRESHOLD = 0.5


def get_category_token_spans(processor) -> list[list[int]]:
    """Token index ranges for each of the 12 category phrases in TEXT_PROMPT,
    found by splitting on the "." token -- same delimiter convention
    transformers' build_label_maps uses internally during training.
    """
    tokenizer = processor.tokenizer
    input_ids = tokenizer(TEXT_PROMPT, return_tensors="pt")["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    spans, current = [], []
    for i, tok in enumerate(tokens):
        if tok in ("[CLS]", "[SEP]"):
            continue
        if tok == ".":
            spans.append(current)
            current = []
        else:
            current.append(i)
    assert len(spans) == len(CATEGORY_NAMES), f"expected {len(CATEGORY_NAMES)} phrase spans, got {len(spans)}"
    return spans


@torch.no_grad()
def evaluate_image(model, processor, spans, image: Image.Image, gt_annotations: list[dict], device: str) -> list:
    """Returns list of (true_category_id, predicted_category_id_or_None) for each GT box."""
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        outputs = model(**inputs)

    probs = outputs.logits.sigmoid()[0].float()  # (num_queries, seq_len)
    query_category_scores = torch.stack([probs[:, span].mean(dim=1) for span in spans], dim=1)  # (num_queries, 12)
    query_best_score, query_best_category = query_category_scores.max(dim=1)

    boxes_xyxy_norm = center_to_corners_format(outputs.pred_boxes[0])
    w, h = image.size
    pred_boxes_xyxy = (boxes_xyxy_norm * torch.tensor([w, h, w, h], device=device)).cpu()

    gt_xyxy = torch.tensor(
        [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in gt_annotations]
    )
    ious = torchvision.ops.box_iou(gt_xyxy, pred_boxes_xyxy)

    results = []
    for gi, ann in enumerate(gt_annotations):
        candidates = (ious[gi] > IOU_MATCH_THRESHOLD).nonzero().flatten()
        if len(candidates) == 0:
            results.append((ann["category_id"], None))
            continue
        best_query = candidates[query_best_score[candidates].argmax()]
        results.append((ann["category_id"], query_best_category[best_query].item()))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "checkpoints" / "final")
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, checkpoint: {args.checkpoint}")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, size=IMAGE_SIZE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(device)
    model.eval()
    spans = get_category_token_spans(processor)

    with open(DATA_DIR / "test.json") as f:
        test = json.load(f)
    anns_by_image: dict[int, list] = {}
    for ann in test["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = test["images"][: args.limit_images] if args.limit_images else test["images"]
    print(f"evaluating on {len(images)} test images")

    correct = {i: 0 for i in range(len(CATEGORY_NAMES))}
    total = {i: 0 for i in range(len(CATEGORY_NAMES))}
    unmatched = {i: 0 for i in range(len(CATEGORY_NAMES))}

    for n, im in enumerate(images):
        gt_annotations = anns_by_image.get(im["id"], [])
        if not gt_annotations:
            continue
        image = Image.open(DATA_DIR / im["path"]).convert("RGB")
        for true_cat, pred_cat in evaluate_image(model, processor, spans, image, gt_annotations, device):
            total[true_cat] += 1
            if pred_cat is None:
                unmatched[true_cat] += 1
            elif pred_cat == true_cat:
                correct[true_cat] += 1

        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(images)} images")

    print("\n=== per piece type accuracy (IoU>0.5 box match, argmax over phrase-span probability) ===")
    overall_correct, overall_total = 0, 0
    for cid, name in enumerate(CATEGORY_NAMES):
        acc = correct[cid] / total[cid] if total[cid] else float("nan")
        print(f"  {name:15s}: {correct[cid]:4d}/{total[cid]:4d} = {acc:.3f}  (no-detection: {unmatched[cid]})")
        overall_correct += correct[cid]
        overall_total += total[cid]

    print(f"\noverall: {overall_correct}/{overall_total} = {overall_correct / overall_total:.3f}")


if __name__ == "__main__":
    main()
