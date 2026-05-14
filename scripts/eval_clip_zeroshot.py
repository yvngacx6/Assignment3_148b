"""§3.3 — Zero-shot qualitative analysis on EuroSAT.

Loads a CLIP-pretrained checkpoint, runs zero-shot classification over the
EuroSAT validation split by default, and writes:

  {output_dir}/correct.png           5-image montage of correct predictions
  {output_dir}/wrong.png             5-image montage of misclassifications
  {output_dir}/correct_examples.json selected correct examples
  {output_dir}/wrong_examples.json   selected wrong examples with top-3 classes
  {output_dir}/confusion_top3.json   the 3 worst-classified true classes,
                                     with their most-common wrong prediction
  {output_dir}/confusion_full.png    10x10 confusion matrix (counts)
  {output_dir}/summary.json          overall zero-shot accuracy

You implement: `predict_all`, `pick_correct_and_wrong`, `top3_mistakes`.
The rest is plumbing.

Usage (typically on Colab, where best.pt already lives):
    uv run python scripts/eval_clip_zeroshot.py \
        --config configs/clip_eurosat.yaml \
        --checkpoint runs/clip_eurosat/best.pt \
        --output-dir figures/clip_eurosat_qualitative
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders


# ---------------------------------------------------------------------------
# Plumbing helpers (provided)
# ---------------------------------------------------------------------------


# ImageNet stats — must match `vlm.data.default_image_transform`.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                   help="YAML config used during training (for ViT geometry).")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to best.pt or last.pt from pretrain_clip.py.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("figures/clip_eurosat_qualitative"))
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="Dataset split to analyze. The writeup asks for validation examples.")
    p.add_argument("--num-correct", type=int, default=5)
    p.add_argument("--num-wrong", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> tuple[ViT, FrozenTextEncoder, ProjectionHeads]:
    """Rebuild the same architecture used during pretraining and load weights."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    vit = ViT(
        img_size=cfg["vit"]["img_size"],
        patch_size=cfg["vit"]["patch_size"],
        d_model=cfg["vit"]["d_model"],
        num_heads=cfg["vit"]["num_heads"],
        num_blocks=cfg["vit"]["num_blocks"],
        dropout=cfg["vit"]["dropout"],
    ).to(device)
    vit.load_state_dict(state["vit"])
    vit.eval()

    text_encoder = FrozenTextEncoder(model_name=cfg["text_encoder"]["model_name"]).to(device)

    proj = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    proj.load_state_dict(state["projection_heads"])
    proj.eval()

    if "meta" in state:
        print(f"Loaded checkpoint meta: {state['meta']}")
    return vit, text_encoder, proj


@torch.no_grad()
def encode_class_embeddings(
    text_encoder: FrozenTextEncoder,
    proj: ProjectionHeads,
    class_names: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Encode each class prompt once and project + normalize.

    Returns:
        class_proj: (num_classes, d_proj) tensor on `device`, L2-normalized.
        prompts:    the prompt strings (same order as class_names).
    """
    prompts = [f"a satellite image of {c}" for c in class_names]
    text_embeds = text_encoder(prompts).clone().to(device)  # .clone() escapes inference-mode
    # We only need text_proj here; pass a zero image tensor of the right shape.
    image_dummy = torch.zeros(len(prompts), proj.image_proj.in_features, device=device)
    _, class_proj = proj(image_dummy, text_embeds)
    class_proj = F.normalize(class_proj, dim=-1)
    return class_proj, prompts


def denormalize(img: torch.Tensor) -> "np.ndarray":
    """Undo ImageNet normalization and return an HxWx3 uint8 array suitable for imshow."""
    import numpy as np
    img = img.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _setup_matplotlib():
    """Same Colab-friendly matplotlib setup used in pretrain_clip.py::save_curves."""
    os.environ.pop("MPLBACKEND", None)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_montage(items: list[dict], class_names: list[str], title: str, path: Path) -> None:
    """Render a horizontal strip of images with per-image captions.

    Each `item` must have keys: 'image' (CHW tensor), 'true_label', 'pred_label'.
    For correct predictions, caption shows just the true class.
    For wrong predictions, caption shows 'true: X | pred: Y'.
    """
    plt = _setup_matplotlib()
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, item in zip(axes, items):
        ax.imshow(denormalize(item["image"]))
        true_name = class_names[item["true_label"]]
        pred_name = class_names[item["pred_label"]]
        if item["true_label"] == item["pred_label"]:
            ax.set_title(true_name, fontsize=9)
        else:
            ax.set_title(f"true: {true_name}\npred: {pred_name}", fontsize=8, color="crimson")
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_matrix_png(
    cm: list[list[int]],
    class_names: list[str],
    path: Path,
) -> None:
    """Heatmap of the 10x10 confusion matrix with per-cell counts."""
    plt = _setup_matplotlib()
    import numpy as np
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_arr / row_sums

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    n = len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm_arr[i, j])),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative-analysis helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_all(
    vit: ViT,
    proj: ProjectionHeads,
    test_loader,
    class_proj: torch.Tensor,
    class_prompts: list[str],
    device: torch.device,
) -> list[dict]:
    """Run the model over the test loader and collect per-image predictions.

    Returns a list with one dict per test image, each containing:
        'image'      : the input image tensor (CHW, on CPU, ImageNet-normalized)
        'true_label' : int in [0, num_classes)
        'pred_label' : int in [0, num_classes)
        'pred_score' : float — the cosine similarity of (img_proj, class_proj[pred_label])
        'all_scores' : list[float] of length num_classes (for the confusion analysis)

    Hints:
      - The test loader yields (images, list[str captions]).
      - Recover true labels exactly the way `vlm.eval.zeroshot_classification_accuracy`
        does: `[class_prompts.index(c) for c in captions]`.
      - For each image: feats = vit(image); img_proj, _ = proj(feats, dummy_text);
        img_proj = F.normalize(img_proj, dim=-1); scores = img_proj @ class_proj.T.
      - To keep memory under control, store images on CPU (`image.cpu()`).

    """
    vit.eval()
    proj.eval()

    predictions: list[dict] = []
    class_proj = class_proj.to(device)

    for images, captions in test_loader:
        images = images.to(device, non_blocking=True)
        labels = torch.tensor(
            [class_prompts.index(caption) for caption in captions],
            device=device,
            dtype=torch.long,
        )

        image_feats = vit(images)
        dummy_text = torch.zeros(
            image_feats.shape[0],
            proj.text_proj.in_features,
            device=device,
        )
        image_proj, _ = proj(image_feats, dummy_text)
        image_proj = F.normalize(image_proj, dim=-1)

        scores = image_proj @ class_proj.T
        pred_scores, pred_labels = scores.max(dim=-1)

        for image, true_label, pred_label, pred_score, all_scores in zip(
            images.cpu(),
            labels.cpu(),
            pred_labels.cpu(),
            pred_scores.cpu(),
            scores.cpu(),
        ):
            predictions.append({
                "image": image,
                "true_label": int(true_label.item()),
                "pred_label": int(pred_label.item()),
                "pred_score": float(pred_score.item()),
                "all_scores": [float(score) for score in all_scores.tolist()],
            })

    return predictions


def pick_correct_and_wrong(
    predictions: list[dict],
    n_correct: int,
    n_wrong: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Pick `n_correct` correctly-classified and `n_wrong` misclassified items.

    A few reasonable strategies — pick one and document it in the writeup:
      - Random sample (most representative; use rng.sample).
      - Highest-confidence correct + highest-confidence WRONG (the wrongs are
        the most interesting because the model was very confidently wrong).
      - Top-class diversity (one per true class) — harder to implement.

    Return order of items in each list is what gets rendered left-to-right in
    the montage, so a deterministic sort can produce nicer figures.

    """
    correct = [p for p in predictions if p["true_label"] == p["pred_label"]]
    wrong = [p for p in predictions if p["true_label"] != p["pred_label"]]

    correct_items = rng.sample(correct, k=min(n_correct, len(correct)))
    wrong_items = rng.sample(wrong, k=min(n_wrong, len(wrong)))

    correct_items.sort(key=lambda p: (p["true_label"], -p["pred_score"]))
    wrong_items.sort(key=lambda p: (p["true_label"], p["pred_label"], -p["pred_score"]))
    return correct_items, wrong_items


def top3_mistakes(
    predictions: list[dict],
    class_names: list[str],
) -> tuple[list[dict], list[list[int]]]:
    """Identify the 3 true classes with the *lowest* per-class accuracy and
    also build the full 10x10 confusion matrix.

    Returns:
        top3:  list of 3 dicts, each:
            {
              'true_class': str,
              'accuracy': float,                      # correct / total for that class
              'support': int,                         # number of test images of this class
              'most_common_wrong_pred': (str, int),   # (predicted class, count)
            }
        confusion_matrix: list[list[int]] of shape (num_classes, num_classes),
                          where cm[i][j] = #(true=i, pred=j).

    Hints:
      - Build the confusion matrix first by iterating predictions once.
      - Per-class accuracy = cm[i][i] / sum(cm[i][:]).
      - For most-common wrong pred for class i: sort {j: cm[i][j] for j != i} by
        count, take the top one. `collections.Counter` makes this clean.

    """
    num_classes = len(class_names)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for pred in predictions:
        true_label = pred["true_label"]
        pred_label = pred["pred_label"]
        cm[true_label][pred_label] += 1

    per_class: list[dict] = []
    for true_label, row in enumerate(cm):
        support = sum(row)
        correct = row[true_label]
        accuracy = correct / support if support else 0.0

        wrong_counts = Counter({
            pred_label: count
            for pred_label, count in enumerate(row)
            if pred_label != true_label and count > 0
        })
        if wrong_counts:
            most_common_label, most_common_count = wrong_counts.most_common(1)[0]
            most_common_wrong_pred = (class_names[most_common_label], most_common_count)
        else:
            most_common_wrong_pred = (None, 0)

        per_class.append({
            "true_class": class_names[true_label],
            "accuracy": accuracy,
            "support": support,
            "most_common_wrong_pred": most_common_wrong_pred,
        })

    per_class.sort(key=lambda item: (item["accuracy"], -item["support"], item["true_class"]))
    return per_class[:3], cm


def examples_for_json(items: list[dict], class_names: list[str]) -> list[dict]:
    """Drop image tensors and keep the labels/scores needed for the writeup table."""
    rows = []
    for item in items:
        top3 = sorted(
            enumerate(item["all_scores"]),
            key=lambda pair: pair[1],
            reverse=True,
        )[:3]
        rows.append({
            "true_class": class_names[item["true_label"]],
            "pred_class": class_names[item["pred_label"]],
            "pred_score": item["pred_score"],
            "top3_predictions": [
                {"class": class_names[label], "score": score}
                for label, score in top3
            ],
        })
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    print(f"Device: {device}")

    _train_loader, val_loader, test_loader = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    eval_loader = val_loader if args.split == "val" else test_loader
    print(f"{args.split} loader: {len(eval_loader)} batches")

    vit, text_encoder, proj = load_model(cfg, args.checkpoint, device)
    class_proj, class_prompts = encode_class_embeddings(
        text_encoder, proj, EUROSAT_CLASSES, device
    )

    predictions = predict_all(vit, proj, eval_loader, class_proj, class_prompts, device)
    n = len(predictions)
    correct = sum(1 for p in predictions if p["true_label"] == p["pred_label"])
    accuracy = correct / max(n, 1)
    print(f"{args.split} zero-shot accuracy: {accuracy:.4f}  ({correct}/{n})")

    rng = random.Random(args.seed)
    correct_items, wrong_items = pick_correct_and_wrong(
        predictions, args.num_correct, args.num_wrong, rng
    )

    save_montage(correct_items, EUROSAT_CLASSES,
                 title=f"Correct ({len(correct_items)}/{args.num_correct})",
                 path=args.output_dir / "correct.png")
    save_montage(wrong_items, EUROSAT_CLASSES,
                 title=f"Misclassified ({len(wrong_items)}/{args.num_wrong})",
                 path=args.output_dir / "wrong.png")

    with open(args.output_dir / "correct_examples.json", "w") as f:
        json.dump(examples_for_json(correct_items, EUROSAT_CLASSES), f, indent=2)
    with open(args.output_dir / "wrong_examples.json", "w") as f:
        json.dump(examples_for_json(wrong_items, EUROSAT_CLASSES), f, indent=2)

    top3, cm = top3_mistakes(predictions, EUROSAT_CLASSES)
    save_confusion_matrix_png(cm, EUROSAT_CLASSES, args.output_dir / "confusion_full.png")

    with open(args.output_dir / "confusion_top3.json", "w") as f:
        json.dump(top3, f, indent=2)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "accuracy": accuracy,
            "num_images": n,
            "num_classes": len(EUROSAT_CLASSES),
        }, f, indent=2)

    print(f"\nWrote artifacts to {args.output_dir}/")
    print(f"Top-3 mistake classes: {[t['true_class'] for t in top3]}")


if __name__ == "__main__":
    main()
