# Assignment Progress

## Assignment metadata

- Assignment name: HW 3 — Vision-Language Models
- Assignment folder: `hw3/`
- Assignment PDF: `homework_pdfs/hw3.pdf`
- Course: EE/CS 148B (Spring 2026)
- Started: 2026-05-09
- Last updated: 2026-05-11
- Current deliverable: §2.4 — written `vit_pooling` (3–4 sentences) and empirical `vit_patch_size` (timing table)

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
| 4  | §2.4 vit_patch_size     | Patch-size sweep wall-clock timing table                         | In progress | new `scripts/bench_patch_size.py` (TBD), `writeup.tex` | timing on `B=16`, P∈{8,16,32}, d=384, H=6, L=6      | mean ± std over 20 steps after 5 warmup |
| 5  | §3.1 clip_setup         | Projection heads (image/text → 256, no bias, L2 norm)            | Not started | `vlm/clip.py`                                         | (covered by clip_loss test indirectly)              | use provided `vlm/data.py`, `basics/text_encoder.py` |
| 6  | §3.2 infonce            | `clip_loss` — symmetric InfoNCE                                  | Not started | `vlm/clip.py`                                         | `uv run pytest -k test_clip_loss`                   | parameterize τ as `exp(logit_scale)`, clamp ≤ ln(100) |
| 7  | §3.3 clip_train         | EuroSAT CLIP pretraining + train-loss & val-acc curves           | Not started | `scripts/pretrain_clip.py`, `configs/clip_eurosat.yaml` | run script, save curves                          | 20 epochs, batch 256, lr 3e-4, AdamW wd=0.1 |
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

#3 — §2.4 `vit_pooling` (written, 3–4 sentences) and #4 — §2.4 `vit_patch_size` (table + 2–3 sentence discussion).

### Status

§2 code (`PatchEmbeddings` and `ViT`) complete; all 4 tests pass. §2.4 has two coupled written/empirical deliverables remaining before §2 is fully done.

### Mode

Waiting on user: confirm compute target (local CUDA vs Colab vs CPU) for the §2.4 timing benchmark, and approve scaffolding for `scripts/bench_patch_size.py`.

### Files relevant to current deliverables

- `hw3/writeup.tex` — both §2.4 answers go here.
- (new) `hw3/scripts/bench_patch_size.py` — wall-clock timing harness for the patch-size sweep. Doesn't exist yet.
- `hw3/basics/vit.py` — used by the timing script (no further edits needed for §2.4).

### Files touched so far

- `hw3/ASSIGNMENT_PROGRESS.md` (this file)
- `hw3/` (cloned from `https://github.com/caltech-eecs148b/hw3`)
- `hw3/writeup.tex` (scaffolded LaTeX template for written deliverables)
- `hw3/basics/vit.py` — `PatchEmbeddings` and `ViT` complete (4/4 tests pass).

### Open TODOs

- Decide compute target for §2.4 timing: local CUDA / Colab / CPU fallback.
- Scaffold `scripts/bench_patch_size.py` (CLI: patch sizes, batch size, warmup, num steps; output: mean ± std table). Will use `torch.cuda.synchronize()` if CUDA available, `time.perf_counter()` otherwise.
- Answer `vit_pooling` (3–4 sentences) in `writeup.tex`.
- Run benchmark, fill `vit_patch_size` table + 2–3 sentence discussion in `writeup.tex`.
- Note for §5 later: deliverable #13 will add an optional `return_all_tokens=True` flag to `ViT.forward`.
- Clean up: drop unused `MultiHeadAttention` import in `basics/vit.py`.

### Tests run

```bash
uv run pytest -k test_patch_embeddings -v   # 2 passed (2026-05-11)
uv run pytest -k test_vit -v                # 4 passed (2026-05-11) — both ViT tests + the 2 patch tests
```

### Waiting on user

1. Where will you run the §2.4 timing benchmark? (local CUDA GPU / Colab / CPU only)
2. Want me to scaffold `scripts/bench_patch_size.py`, or attempt cold? (For the written `vit_pooling` answer I'll wait until you've thought about it; we can discuss before you write it down.)
