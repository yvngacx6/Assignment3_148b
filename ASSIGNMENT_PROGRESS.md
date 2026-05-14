# Assignment Progress

## Assignment metadata

- Assignment name: HW 3 — Vision-Language Models
- Assignment folder: `hw3/`
- Assignment PDF: `homework_pdfs/hw3.pdf`
- Course: EE/CS 148B (Spring 2026)
- Started: 2026-05-09
- Last updated: 2026-05-13
- Current deliverable: §3.3 — `clip_zeroshot` (qualitative EuroSAT zero-shot analysis)

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
| 7  | §3.3 clip_train         | EuroSAT CLIP pretraining + train-loss & val-acc curves           | DONE        | `scripts/pretrain_clip.py`, `configs/clip_eurosat.yaml`, `figures/clip_eurosat/*`, `writeup.tex` §2.3 | Colab run complete; curves saved; writeup populated | best val acc 91.77% at epoch 18; final 91.71% |
| 8  | §3.3 clip_zeroshot      | Qualitative: 5 correct + 5 wrong + top-3 mistakes                | In progress | `scripts/eval_clip_zeroshot.py`, `writeup.tex` §2.4 | reuse trained checkpoint; run qualitative helper     | helper implemented; needs generated artifacts + discussion |
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

#8 — §3.3 `clip_zeroshot` (qualitative zero-shot analysis using the trained EuroSAT CLIP checkpoint).

### Status

§2 fully closed (4/4 tests). §3.1 (`ProjectionHeads`) and §3.2 (`clip_loss`) both implemented and approved (4/4 `test_clip_loss` tests pass; writeup §2.2 InfoNCE rationale done). §3.3 `clip_train` is complete: the training loop ran for 20 epochs, the loss/validation curves are saved under `figures/clip_eurosat/`, and `writeup.tex` §2.3 now contains both figures plus the 2–3 sentence interpretation. `scripts/eval_clip_zeroshot.py` now implements the qualitative prediction/sampling/confusion helpers and writes selected-example JSON for the writeup table.

### Mode

Waiting on user: run `scripts/eval_clip_zeroshot.py` against the trained checkpoint, inspect the generated correct/wrong montages and JSON, then populate `writeup.tex` §2.4.

### Files relevant to current deliverable

- `hw3/scripts/pretrain_clip.py` — training/eval workflow used to produce the checkpoint and curves.
- `hw3/scripts/eval_clip_zeroshot.py` — qualitative helper; runs validation zero-shot, saves correct/wrong montages, top-3 selected-example JSON, and confusion summaries.
- `hw3/notebooks/colab_runner.ipynb` — Step 9 runs the qualitative helper on Colab and displays/stages generated artifacts.
- `hw3/figures/clip_eurosat/train_loss.png` — §3.3 training-loss curve embedded in `writeup.tex`.
- `hw3/figures/clip_eurosat/val_accuracy.png` — §3.3 zero-shot validation accuracy curve embedded in `writeup.tex`.
- `hw3/figures/clip_eurosat/curves.json` — raw curve data; best val acc 91.77% at epoch 18.
- `hw3/writeup.tex` — §2.3 populated; §2.4 qualitative zero-shot analysis remains.

### Files touched so far

- `hw3/ASSIGNMENT_PROGRESS.md` (this file)
- `hw3/` (cloned from `https://github.com/caltech-eecs148b/hw3`)
- `hw3/writeup.tex` (§1.1–§1.4 + §2.2 InfoNCE rationale populated)
- `hw3/basics/vit.py` — `PatchEmbeddings` and `ViT` complete (4/4 tests pass)
- `hw3/scripts/bench_patch_size.py` — §2.4 benchmark complete
- `hw3/vlm/clip.py` — `ProjectionHeads`, `clip_loss`, `init_logit_scale` complete (4/4 tests pass)
- `hw3/scripts/pretrain_clip.py` — §3.3 training loop complete
- `hw3/scripts/eval_clip_zeroshot.py` — §3.3 qualitative helper complete
- `hw3/notebooks/colab_runner.ipynb` — Step 9 qualitative Colab runner added
- `hw3/figures/clip_eurosat/` — §3.3 curves from Colab run

### Open TODOs

- Run Step 9 in `notebooks/colab_runner.ipynb` on Colab, or run `uv run python scripts/eval_clip_zeroshot.py --config configs/clip_eurosat.yaml --checkpoint runs/clip_eurosat/best.pt --output-dir figures/clip_eurosat_qualitative`.
- Review `figures/clip_eurosat_qualitative/correct.png`, `wrong.png`, and `wrong_examples.json`.
- Embed qualitative examples and a 3–4 sentence interpretation in `writeup.tex` §2.4.
- Note for §5 later: deliverable #13 will add an optional `return_all_tokens=True` flag to `ViT.forward`.
- Clean up: drop unused `MultiHeadAttention` import in `basics/vit.py` (if still present).

### Tests run

```bash
uv run pytest -k test_patch_embeddings -v   # 2 passed (2026-05-11)
uv run pytest -k test_vit -v                # 4 passed (2026-05-11)
uv run pytest -k test_clip_loss -v          # 4 passed (2026-05-13)
# Colab T4 §2.4 benchmark completed 2026-05-12; numbers in writeup.tex §1.4 Table 1.
# Colab §3.3 CLIP pretraining completed 2026-05-13; best val acc 91.77% at epoch 18.
```

### Waiting on user

Run the qualitative zero-shot helper against `runs/clip_eurosat/best.pt`, then ping me to help populate the qualitative figure/table and discussion in `writeup.tex`.
