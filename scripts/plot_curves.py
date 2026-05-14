"""Re-plot CLIP pretraining curves from a curves.json dump.

Useful when save_curves crashed mid-run, or when you want to re-render figures
without re-training. Reads `--curves` and writes
`{output_dir}/{train_loss.png, val_accuracy.png}` next to it (or to
`--output-dir` if you want a different destination).

Usage:
    uv run python scripts/plot_curves.py --curves figures/clip_eurosat/curves.json
    uv run python scripts/plot_curves.py --curves runs/clip_eurosat/curves.json \
        --output-dir figures/clip_eurosat
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--curves", type=Path, required=True, help="Path to curves.json.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write PNGs (default: same dir as curves.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.curves.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = json.loads(args.curves.read_text())
    train_losses = curves["train_losses"]
    val_accs = curves["val_accs"]

    # Colab injects MPLBACKEND="module://matplotlib_inline.backend_inline" into
    # subprocess envs; clear it before importing matplotlib (same fix as in
    # scripts/pretrain_clip.py::save_curves).
    os.environ.pop("MPLBACKEND", None)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    best_idx = max(range(len(val_accs)), key=lambda i: val_accs[i])
    print(f"final train loss : {train_losses[-1]:.4f}")
    print(f"final val acc    : {val_accs[-1]:.4f}")
    print(f"best  val acc    : {val_accs[best_idx]:.4f}  (epoch {best_idx + 1})")
    print(f"wrote PNGs to    : {output_dir}/")


if __name__ == "__main__":
    main()
