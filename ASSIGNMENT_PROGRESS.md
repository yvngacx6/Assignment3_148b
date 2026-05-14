# Assignment Progress

## Assignment metadata

- Assignment name: HW 3 — Vision-Language Models
- Assignment folder: `hw3/`
- Assignment PDF: `homework_pdfs/hw3.pdf`
- Course: EE/CS 148B (Spring 2026)
- Started: 2026-05-09
- Last updated: 2026-05-13
- Current deliverable: §3.3 — `clip_train` (EuroSAT pretraining loop in `scripts/pretrain_clip.py`)

## Workflow contract

The agent must help me one deliverable at a time.

For each deliverable:
1. Scaffold only.
2. Leave key logic as TODOs.
3. Wait for me to attempt.
4. Review my attempt.
5. Debug only what is necessary.
6. Move on only when I explicitly approve.

## Deliverable checklist

| # | Section | Deliverable | Status | Files | Tests / Commands | Notes |
|---|---------|-------------|--------|-------|------------------|-------|
| 1  | §2.2 patch_embeddings   | `PatchEmbeddings` module                                         | DONE        | `basics/vit.py`                                       | `uv run pytest -k test_patch_embeddings` ✓ 2/2      | strided Conv2d → flatten → transpose |
| 2  | §2.3 vit                | `ViT` module (CLS + pos embed + Block stack + final LN)          | DONE        | `basics/vit.py`                                       | `uv run pytest -k test_vit` ✓ 2/2                   | use `basics.model.Block(is_decoder=False, block_size=N+1)` |
| 3  | §2.4 vit_pooling        | Written: CLS vs mean pool vs attention pool                      | DONE        | `writeup.tex` §1.3                                    | —                                                   | 4 sentences, approved 2026-05-11 |
| 4  | §2.4 vit_patch_size     | Patch-size sweep wall-clock timing table                         | DONE        | `scripts/bench_patch_size.py`, `writeup.tex` §1.4     | Tesla T4 numbers populated; analytical + discussion in §1.4 | 15.4× measured vs 256× pure-attention prediction |
| 5  | §3.1 clip_setup         | Projection heads (image/text → 256, no bias, L2 norm)            | DONE        | `vlm/clip.py`                                         | (no direct pytest; visual review approved 2026-05-13) | uses `F.normalize(..., dim=-1)` for safe L2 |
| 6  | §3.2 infonce            | `clip_loss` — symmetric InfoNCE                                  | DONE        | `vlm/clip.py`, `writeup.tex` §2.2                     | `pytest -k test_clip_loss` ✓ 4/4                    | both code + 2-sentence rationale approved 2026-05-13 |
| 7  | §3.3 clip_train         | EuroSAT CLIP pretraining + train-loss & val-acc curves           | In progress | `scripts/pretrain_clip.py`, `configs/clip_eurosat.yaml` | run script, save curves                          | 20 epochs, batch 256, lr 3e-4, AdamW wd=0.1 |
| 8  | §3.3 clip_zeroshot      | Qualitative: 5 correct + 5 wrong + top-3 mistakes                | Not started | `scripts/pretrain_clip.py` (or notebook)             | reuse trained checkpoint                            | 3–4 sentence discussion |
| 9  | §4.1 lora_linear        | `LoRALinear` + `apply_lora_to_attention`                         | Not started | `basics/lora.py`                                      | `pytest -k test_lora_linear`, `pytest -k test_apply_lora` | print total/trainable/ratio at rank 8 |
| 10 | §4.2 lora_compare       | RESISC45: linear probe vs LoRA r=8 vs full FT                    | Not started | `scripts/finetune_resisc.py`, `configs/lora_resisc.yaml` | 10 epochs each                                  | report acc, trainable params, peak mem, wall-clock |
| 11 | §4.2 lora_rank          | LoRA rank sweep r ∈ {1,2,4,8,16,32,64}                           | Not started | `scripts/finetune_resisc.py`, `configs/lora_resisc.yaml` | 10 epochs per rank, α=2r                        | accuracy-vs-rank plot |
| 12 | §5.3 projector          | `VisionLanguageProjector` (2-layer MLP, GELU)                    | Not started | `vlm/projector.py`                                    | (used by §5 train script)                           | handle (B, d) and (B, N, d) inputs |
| 13 | §5.4 injection          | `VisionLanguageModel.forward` with 3 injection modes + label shift | Not started | `vlm/model.py`, extend `basics/vit.py` (`return_all_tokens`) | (covered by §5 training)                  | mask visual + non-answer tokens with -100 |
| 14 | §5.4 injection_compare  | Train w/ each strategy, 2000 steps; table                        | Not started | `scripts/train_vlm.py`, `configs/vlm_clevr.yaml`     | val acc, # visual tokens, peak mem, wall-clock      | freeze ViT + decoder, train projector only |
| 15 | §5.5 masking            | Causal vs image-bidir mask + 500-step ablation                   | Not started | `vlm/model.py`, `vlm/masking.py` (provided)          | `scripts/train_vlm.py`                              | also: draw 7×7 diagrams in writeup |
| 16 | §5.6 freezing           | 4-row freeze-strategy table (1500 steps each)                    | Not started | `scripts/train_vlm.py`                                | A: proj only / B: proj+LoRA-decoder / C: proj+full / D: all FT | LoRA decoder must wrap SmolLM2 q/v_proj directly |
| 17 | §5.7 vlm_qualitative    | 10 CLEVR examples + failure-mode discussion                      | Not started | `scripts/eval_vlm.py`                                 | reuse best checkpoint                               | hypothesize encoder-vs-decoder failures |
| 18 | §6.1 rope_1d            | `RoPE1D` module + norm-preservation check                        | Not started | `basics/rope.py`                                      | `uv run pytest -k test_rope_1d`                     | precompute cos/sin as buffers |
| 19 | §6.1 rope_vs_learned    | Learned PE vs 1D RoPE on EuroSAT + length-extrapolation test     | Not started | `basics/vit.py`, `scripts/pretrain_clip.py`          | retrain each 20 epochs                              | eval at 96×96 (12×12 grid); interpolate learned PE |
| 20 | §6.2 rope_2d            | `RoPE2D` + EuroSAT ablation                                      | Not started | `basics/rope.py`, `basics/vit.py`, `scripts/pretrain_clip.py` | `pytest -k test_rope_2d`                  | head_dim divisible by 4 |
| 21 | §6.3 mrope_written      | 3-paragraph M-RoPE writeup                                       | Not started | writeup only                                          | —                                                   | naive PE issues / first-text-token pos / why 3 chunks |
| 22 | §6.3 mrope_impl (BONUS) | M-RoPE position assignment + CLEVR ablation (1500 steps)         | Not started | `vlm/model.py`, `scripts/train_vlm.py`, `scripts/eval_vlm.py` | overall + spatial-question accuracy        | optional bonus |

## Current state

### Current deliverable

#7 — §3.3 `clip_train` (EuroSAT CLIP pretraining loop in `scripts/pretrain_clip.py`).

### Status

§2 fully closed (4/4 tests). §3.1 (`ProjectionHeads`) and §3.2 (`clip_loss`) both implemented and approved (4/4 `test_clip_loss` tests pass; writeup §2.2 InfoNCE rationale done). §3.3 scaffold landed in `scripts/pretrain_clip.py` 2026-05-13: plumbing (config load, seed, model build, AdamW, cosine+warmup LambdaLR, zero-shot eval wrapper, curve plotting, checkpoint helpers) is complete; **three TODO blocks remain** for the student to fill in.

### Mode

Waiting on user: fill the three `TODO(student)` blocks in `scripts/pretrain_clip.py`, then run the script (likely on Colab L4) and capture loss + zero-shot val accuracy curves.

### Files relevant to current deliverable

- `hw3/scripts/pretrain_clip.py` — three open TODOs:
  1. **CLIP training step** inside `train_one_epoch` (encode → project → `clip_loss` → backward → step → scheduler → `logit_scale.data.clamp_(max=ln 100)` → record loss).
  2. **Best-checkpoint rule** inside `main` (compare `val_acc > best_acc`, update, call `save_checkpoint(args.output_dir / "best.pt", ...)`).
  3. (Implicit, inside #1): the `logit_scale` clamp — easy to forget; the `clip_loss` docstring explicitly delegates this to the training loop.
- `hw3/configs/clip_eurosat.yaml` — hyperparams (img_size 64, patch 8, d=384, 6 heads, 6 blocks, dropout 0.1; AdamW lr 3e-4 wd 0.1 betas (0.9, 0.95); 200-step warmup + cosine; 20 epochs, batch 256).
- `hw3/vlm/data.py::build_eurosat_loaders` — yields `(images, list[str captions])` per batch.
- `hw3/vlm/clip.py` — `ProjectionHeads`, `clip_loss`, `init_logit_scale` (all done).
- `hw3/basics/text_encoder.py` — `FrozenTextEncoder(captions: list[str]) -> (B, embed_dim)`.
- `hw3/vlm/eval.py::zeroshot_classification_accuracy` — provided; called from `evaluate_zeroshot` wrapper.

### Files touched so far

- `hw3/ASSIGNMENT_PROGRESS.md` (this file)
- `hw3/` (cloned from `https://github.com/caltech-eecs148b/hw3`)
- `hw3/writeup.tex` (§1.1–§1.4 + §2.2 InfoNCE rationale populated)
- `hw3/basics/vit.py` — `PatchEmbeddings` and `ViT` complete (4/4 tests pass)
- `hw3/scripts/bench_patch_size.py` — §2.4 benchmark complete
- `hw3/vlm/clip.py` — `ProjectionHeads`, `clip_loss`, `init_logit_scale` complete (4/4 tests pass)
- `hw3/scripts/pretrain_clip.py` — §3.3 scaffold landed (2026-05-13); 3 TODOs remain

### Open TODOs

- Fill the three `TODO(student)` blocks in `scripts/pretrain_clip.py`.
- Run `uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml` on Colab (L4 recommended for §3) and capture the two PNGs + final val acc.
- Embed both curves and a 1–2 sentence interpretation in `writeup.tex` §2.3.
- Note for §5 later: deliverable #13 will add an optional `return_all_tokens=True` flag to `ViT.forward`.
- Clean up: drop unused `MultiHeadAttention` import in `basics/vit.py` (if still present).

### Tests run

```bash
uv run pytest -k test_patch_embeddings -v   # 2 passed (2026-05-11)
uv run pytest -k test_vit -v                # 4 passed (2026-05-11)
uv run pytest -k test_clip_loss -v          # 4 passed (2026-05-13)
# Colab T4 §2.4 benchmark completed 2026-05-12; numbers in writeup.tex §1.4 Table 1.
```

### Waiting on user

Attempt the three `TODO(student)` blocks in `scripts/pretrain_clip.py`. When you're done, ping me to review before kicking off the 20-epoch Colab run.
