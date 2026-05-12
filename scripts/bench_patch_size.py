"""§2.4 patch-size sweep wall-clock benchmark.

Implements the timing experiment specified by Problem (vit_patch_size):
  - Image size 224x224, batch size 16
  - ViT config: d_model=384, num_heads=6, num_blocks=6, dropout=0.0
  - Patch sizes P in {8, 16, 32}  ->  N in {784, 196, 49}
  - 5 warmup steps + 20 timed steps per patch size
  - Reports mean +/- std of forward-pass wall-clock time in ms

Usage:
    # default sweep
    python scripts/bench_patch_size.py

    # custom sweep
    python scripts/bench_patch_size.py --patch-sizes 4 8 16 32 --warmup-steps 10

This script is device-aware:
  - On CUDA, it uses torch.cuda.synchronize() to get accurate wall-clock
    (CUDA kernels are launched asynchronously, so naive time.perf_counter()
    measures *launch* time rather than *completion* time).
  - On CPU, CUDA sync is a no-op; time.perf_counter() alone suffices.
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

import torch

from basics.vit import ViT


# -----------------------------------------------------------------------------
# Boilerplate (provided)
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-size", type=int, default=224, help="Image side length (square).")
    parser.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Patch sizes to sweep. img_size must be divisible by each.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model(args: argparse.Namespace, patch_size: int, device: torch.device) -> ViT:
    model = ViT(
        img_size=args.img_size,
        patch_size=patch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=0.0,
    )
    return model.to(device).eval()


def cuda_sync(device: torch.device) -> None:
    """No-op on CPU, full sync on CUDA. Use this around your timed block."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# Timing protocol (TODOs for you)
# -----------------------------------------------------------------------------


def time_one_forward(model: ViT, x: torch.Tensor, device: torch.device) -> float:
    """Time a single forward pass and return the latency in seconds.

    The canonical CUDA-safe timing recipe is:
        1) cuda_sync(device)              # flush any prior async work
        2) start = time.perf_counter()
        3) <run the thing you're timing>
        4) cuda_sync(device)              # wait for the kernel(s) to finish
        5) end = time.perf_counter()
        6) return end - start

    Step 4 is the one that catches people. Without it, perf_counter() measures
    only the time to *launch* the CUDA kernel, not the time for the GPU to
    actually compute the result. You'd get suspiciously fast (and meaningless)
    numbers.

    TODO (you): implement the 6 steps above.
    """
    raise NotImplementedError


def benchmark_one(args: argparse.Namespace, patch_size: int, device: torch.device) -> dict:
    """Run warmup + timed steps for a single patch size and return a result row."""
    torch.manual_seed(args.seed)
    model = build_model(args, patch_size, device)
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    num_patches = (args.img_size // patch_size) ** 2

    # TODO (you): wrap everything below in `torch.no_grad()`. The writeup
    # specifies forward-only timing; without no_grad PyTorch builds the
    # autograd graph for every forward pass, which adds memory + a small
    # amount of compute overhead and is NOT representative of inference time.

    # TODO (you): run args.warmup_steps untimed forward passes of model(x).
    #
    # Why warmup matters:
    #   - On CUDA, the first 1-2 forward passes pay a one-time cost for
    #     kernel compilation, autotuning of cuBLAS GEMM algorithms, and
    #     priming of memory allocator caches. These are 2x-10x slower than
    #     steady state and would skew the mean badly if included.
    #   - On CPU, the first pass often pays an MKL/OpenMP thread-pool
    #     spin-up cost, less dramatic but still real.
    #   - Don't time these; just call model(x) and discard the output.

    # TODO (you): run args.timed_steps timed forward passes. Collect each
    # call's latency in seconds into a list (use `time_one_forward` above).

    # TODO (you): compute the mean and std of those latencies, converted
    # to milliseconds. The `statistics.mean` and `statistics.stdev` helpers
    # imported at the top are fine; or use a NumPy/torch one-liner.
    mean_ms: float = ...   # TODO (you): replace `...` with your value
    std_ms: float = ...    # TODO (you): replace `...` with your value

    return {
        "patch_size": patch_size,
        "num_patches": num_patches,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
    }


# -----------------------------------------------------------------------------
# Entry point (provided)
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(
        f"Config: img_size={args.img_size}, batch_size={args.batch_size}, "
        f"d_model={args.d_model}, num_heads={args.num_heads}, num_blocks={args.num_blocks}"
    )
    print(f"Protocol: {args.warmup_steps} warmup + {args.timed_steps} timed steps per patch size")

    results: list[dict] = []
    for p in args.patch_sizes:
        assert args.img_size % p == 0, f"img_size {args.img_size} not divisible by patch size {p}"
        print(f"\n--- Patch size P = {p} ---")
        result = benchmark_one(args, p, device)
        results.append(result)
        print(
            f"  N = {result['num_patches']}    "
            f"mean = {result['mean_ms']:.3f} ms    "
            f"std = {result['std_ms']:.3f} ms"
        )

    # Final summary table — copy these numbers straight into writeup.tex.
    print("\n=== Summary (copy into writeup.tex Table 1) ===")
    print(f"{'P':>4} | {'N':>6} | {'mean (ms)':>12} | {'std (ms)':>10}")
    print("-" * 42)
    for r in results:
        print(
            f"{r['patch_size']:>4} | {r['num_patches']:>6} | "
            f"{r['mean_ms']:>12.3f} | {r['std_ms']:>10.3f}"
        )


if __name__ == "__main__":
    main()
