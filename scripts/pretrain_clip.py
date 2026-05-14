"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, optimizer + cosine-warmup schedule, zero-shot evaluation hook,
checkpoint saving, and curve plotting.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
from vlm.eval import zeroshot_classification_accuracy


# ---------------------------------------------------------------------------
# Plumbing helpers (provided)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    """LR multiplier: linear ramp 0 -> 1 over `warmup_steps`, then cosine decay
    1 -> 0 across the remaining steps. Returned callable takes the global step
    and returns the multiplier."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def build_model(cfg: dict, device: torch.device) -> tuple[ViT, FrozenTextEncoder, ProjectionHeads, nn.Parameter]:
    vit = ViT(
        img_size=cfg["vit"]["img_size"],
        patch_size=cfg["vit"]["patch_size"],
        d_model=cfg["vit"]["d_model"],
        num_heads=cfg["vit"]["num_heads"],
        num_blocks=cfg["vit"]["num_blocks"],
        dropout=cfg["vit"]["dropout"],
    ).to(device)

    text_encoder = FrozenTextEncoder(model_name=cfg["text_encoder"]["model_name"]).to(device)

    proj = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)

    logit_scale = init_logit_scale()
    # `nn.Parameter.to(device)` returns a non-leaf tensor when the device
    # actually changes, which AdamW rejects. Move the underlying storage
    # in-place so leaf-ness (and therefore optimizer eligibility) is preserved.
    logit_scale.data = logit_scale.data.to(device)
    return vit, text_encoder, proj, logit_scale


def build_optimizer(
    vit: ViT, proj: ProjectionHeads, logit_scale: nn.Parameter, cfg: dict
) -> AdamW:
    """AdamW over the trainable parameters: ViT + projection heads + logit_scale.
    The frozen text encoder is excluded automatically (its params have
    requires_grad=False)."""
    trainable = [p for p in vit.parameters() if p.requires_grad]
    trainable += [p for p in proj.parameters() if p.requires_grad]
    trainable += [logit_scale]
    return AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )


def save_curves(train_losses: list[float], val_accs: list[float], output_dir: Path) -> None:
    """Write loss + accuracy PNGs and a JSON dump of the raw numbers."""
    # Colab injects MPLBACKEND="module://matplotlib_inline.backend_inline" into
    # subprocess envs. That backend only works inside a Jupyter kernel; recent
    # matplotlib versions reject it at import time when run from plain python.
    # Clear the env var and force the headless 'Agg' backend before any
    # matplotlib import.
    import os
    os.environ.pop("MPLBACKEND", None)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title("CLIP InfoNCE loss")
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(val_accs) + 1), val_accs, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val zero-shot accuracy")
    ax.set_title("EuroSAT zero-shot")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "val_accuracy.png", dpi=120)
    plt.close(fig)

    with open(output_dir / "curves.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_accs": val_accs}, f)


def save_checkpoint(
    path: Path,
    vit: ViT,
    proj: ProjectionHeads,
    logit_scale: nn.Parameter,
    extra: dict | None = None,
) -> None:
    state = {
        "vit": vit.state_dict(),
        "projection_heads": proj.state_dict(),
        "logit_scale": logit_scale.detach().cpu(),
    }
    if extra:
        state["meta"] = extra
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    vit: ViT,
    text_encoder: FrozenTextEncoder,
    proj: ProjectionHeads,
    logit_scale: nn.Parameter,
    loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    log_every: int,
    train_losses: list[float],
) -> None:
    vit.train()
    proj.train()
    for step, (images, captions) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        # ------------------------------------------------------------------
        # TODO(student): one CLIP training step.
        #
        #   1. Encode images:  feats_img = vit(images)            # (B, d_image)
        #   2. Encode captions with the FROZEN text encoder
        #      (it expects a python list[str] and returns (B, d_text)).
        #      Make sure the result lives on `device`.
        #   3. Project + L2-normalize via `proj(feats_img, feats_text)`
        #      -> (img_proj, text_proj), each (B, d_proj).
        #   4. Compute loss with `clip_loss(img_proj, text_proj, logit_scale)`.
        #   5. Standard backprop:  zero_grad -> backward -> step.
        #   6. Step the scheduler EACH STEP (cosine schedule is per-step).
        #   7. Clamp logit_scale to <= ln(100) AFTER optimizer.step():
        #         with torch.no_grad():
        #             logit_scale.data.clamp_(max=math.log(100.0))
        #   8. Record the scalar loss into `train_losses`.
        # ------------------------------------------------------------------
        
        # get img embeddings
        feats_img = vit(images)

        # get text embeddings
        # `.clone()` escapes the inference-mode tagging that
        # sentence-transformers' .encode() applies to its outputs; without it,
        # the linear projection below cannot save its input for backward.
        feats_text = text_encoder(captions).clone()

        # project and normalize
        img_proj, text_proj = proj(feats_img, feats_text)

        # compute loss
        loss = clip_loss(img_proj, text_proj, logit_scale)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step scheduler
        scheduler.step()

        # clamp logit scale
        with torch.no_grad():
            logit_scale.data.clamp_(max=math.log(100.0))

        # record loss
        train_losses.append(loss.item())

        if step % log_every == 0:
            print(
                f"  epoch {epoch:02d} step {step:04d}/{len(loader):04d}  "
                f"loss={train_losses[-1]:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"logit_scale={logit_scale.item():.3f}"
            )


@torch.no_grad()
def evaluate_zeroshot(
    vit: ViT,
    text_encoder: FrozenTextEncoder,
    proj: ProjectionHeads,
    val_loader,
    device: torch.device,
) -> float:
    """Wrap the provided helper. The eval helper recovers labels by matching
    each batch caption against `class_prompts`, so the prompt template here MUST
    match the one used by EuroSATCLIPDataset ('a satellite image of {name}')."""
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))
    return zeroshot_classification_accuracy(
        vit=vit,
        projection_heads=proj,
        text_encoder=text_encoder,
        val_loader=val_loader,
        class_prompts=class_prompts,
        class_indices=class_indices,
        device=device,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader, _test_loader = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    print(f"Loaders: train={len(train_loader)} batches, val={len(val_loader)} batches")

    vit, text_encoder, proj, logit_scale = build_model(cfg, device)
    optimizer = build_optimizer(vit, proj, logit_scale, cfg)

    total_steps = cfg["train"]["num_epochs"] * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=cosine_warmup_lambda(cfg["optim"]["warmup_steps"], total_steps),
    )

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_acc = -1.0

    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        t0 = time.time()
        train_one_epoch(
            vit=vit,
            text_encoder=text_encoder,
            proj=proj,
            logit_scale=logit_scale,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_every=cfg["train"]["log_every"],
            train_losses=train_losses,
        )
        train_secs = time.time() - t0

        if epoch % cfg["train"]["eval_every_epoch"] == 0:
            val_acc = evaluate_zeroshot(vit, text_encoder, proj, val_loader, device)
            val_accs.append(val_acc)
            print(
                f"epoch {epoch:02d}  train_secs={train_secs:.1f}  "
                f"val_acc={val_acc:.4f}  best={max(best_acc, val_acc):.4f}"
            )

            # ----------------------------------------------------------------
            # TODO(student): best-checkpoint rule.
            #
            #   If val_acc improves on best_acc, update best_acc and save the
            #   model to args.output_dir / "best.pt" via save_checkpoint(...).
            #   Pass extra={"epoch": epoch, "val_acc": val_acc}.
            # ----------------------------------------------------------------

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(args.output_dir / "best.pt", vit, proj, logit_scale, extra={"epoch": epoch, "val_acc": val_acc})

    save_checkpoint(
        args.output_dir / "last.pt",
        vit, proj, logit_scale,
        extra={"epoch": cfg["train"]["num_epochs"]},
    )
    save_curves(train_losses, val_accs, args.output_dir)
    print(f"Wrote curves and checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
