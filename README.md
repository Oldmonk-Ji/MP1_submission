# DASE7506 MP1 — Small Language Model Challenge

Frozen final submission (2026-09-29): **test BPB 1.414029** (validation 1.3947; baseline ≈ 2.10, **−0.69**), scored in FP32 on CPU.
Evaluation budget: 31.61 s test CPU FP32 (4.14× same-session baseline 7.63 s, clean idle), 0.69 GiB peak RAM, 42.5 MiB inference assets
(limits: 5×, 4 GiB, 64 MiB). Method: knowledge distillation from a nine-member logit ensemble
("super-teacher v3") into a single 10.55M-parameter modern-architecture transformer (RoPE, RMSNorm,
SwiGLU, bias-free linears, zero-init output projections), closed by low-lr WSD continuation passes, then a
fifth round: global temperature calibration T=1.08 on frozen weights, two 2400-step short KD continuations
(lr 0.001), and weight averaging of the two branches (validation 1.3947, Section 2.2 of REPORT.pdf). A
post-freeze CPU inference fast path (fused attention and elementwise kernels, more intra-op threads) keeps
scoring inside the time budget without touching the weights.

- [REPORT.pdf](REPORT.pdf) — the report, LaTeX-rendered (method, comparisons, ablations, critical analysis; 8 pages)
- [GUIDE.md](GUIDE.md) — the assignment guide
- [code/README.md](code/README.md) — implementation notes, measurement procedures, data attribution

## AI assistance disclosure

Substantive AI assistance (Anthropic Claude Code) was used for: implementation scaffolding of the student
model, training-recipe parameterization, the neural-cache variants, ensemble/averaging tooling, experiment
orchestration, and drafting assistance for the report. All experimental design decisions, hyperparameter
choices, result analyses, and final text were reviewed and owned by the author. Reused code: KellerJordan's
`muon.py` (MIT license, unmodified; see `code/tools/muon.py`).

The 2026-09-29 development experiments additionally used OpenAI Codex for checkpoint inspection, calibration runs, a controlled KD-temperature implementation, verification, and experiment documentation.

## Reproduction

### Environment

Python 3.12, torch 2.7.1 (CUDA 12.6 build used for training), numpy 2.1.3, tokenizers 0.21.4, numba 0.61.2
(`code/requirements.txt`). The development environment was a conda env named `mp1`.

```bash
pip install -r code/requirements.txt
cd code
python -m unittest discover -s tests -v   # contract tests, 7/7 pass
```

### Frozen evaluation (no retraining)

The checkpoint bundle is distributed separately via the course submission form. Place the final checkpoint
at the path below and run the unchanged evaluator:

```bash
python evaluate.py --checkpoint runs/p2-avg2-2400/calib-grid/best.pt \
  --device cpu --precision fp32 --split test
```

Expected: test BPB **1.414029** (31.61 s on the submission machine, clean idle; same-session baseline 7.63 s,
4.14× of the 5× limit). The shipped `student.py` includes a CPU FP32 inference fast path (fused numba
kernels for attention and FFN/norm elementwise ops, 12 intra-op threads), which engages only in evaluation
mode; the printed BPB is unchanged by it.
Checkpoint SHA-256 `aca6579fc009ca3e75b8a406249dc175004a29b6e1a93fb0cf9512db10127608`
(10.55M params; modern architecture: RoPE, RMSNorm, SwiGLU, bias-free, zero-init; single forward pass at
scoring, T=1.08).

### Full training chain

All runs: GPU BF16, `--implementation student`. Data, tokenizer and evaluator are unchanged (hashes in
`code/data/manifest.json`); all development and selection used validation only (CPU FP32), and test was
scored only after freezing the method. Total training ≈ 31 GPU-hours (RTX A2000 Laptop).

```bash
# members (GPU, ~11/20 min each)
python train.py --implementation student --config configs/student_v2_backout.json --device cuda --precision bf16 \
  --seed 17 --steps 9600 --cycles 4 --optimizer muon --lr 0.02 --aux-lr-scale 1 \
  --weight-decay 0.01 --label-smoothing 0.1 --ema-decay 0.999 --eval-every 1200 --run-dir runs/v2-muback
python train.py --implementation student --config configs/student_v4_backout.json --device cuda --precision bf16 \
  --seed 17 --steps 9600 --cycles 4 --optimizer muon --lr 0.02 --aux-lr-scale 1 \
  --weight-decay 0.01 --label-smoothing 0.1 --ema-decay 0.999 --eval-every 1200 --run-dir runs/v4-backout

# first submission's predictor (no retraining needed — use the submitted bundle)
python tools/make_ensemble.py runs/v2-muback/ema.pt runs/v4-backout/ema.pt -o ensemble.pt --weights 0.4 0.6

# first submission's frozen evaluation (CPU, FP32)
python evaluate.py --checkpoint ensemble.pt --device cpu --precision fp32 --split test

# second submission's distilled model (GPU, ~43 min). Teacher = seed-17 288x10 (1.4251), itself reached by applying the same
# command at each stage of the Report's Section 2.7 table (student config per stage, teacher = previous stage's ema_0.998).
python train.py --implementation student --config configs/student_288x10x9_backout.json \
  --teacher-checkpoint runs/p0-kd288x10-9600-a075t3-s17/ema_0.998.pt --kd-alpha 0.75 --kd-temperature 3 \
  --device cuda --precision bf16 --seed 42 --steps 9600 --schedule sgdr --cycles 2 \
  --optimizer muon --lr 0.02 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kd288x10-9600-a075t3-s42

# second submission's frozen evaluation (CPU, FP32)
python evaluate.py --checkpoint runs/p0-kd288x10-9600-a075t3-s42/ema_0.998.pt --device cpu --precision fp32 --split test

# --- third round ---
# super-teacher: 5-member ensemble, weights/temperature tuned on validation
python tools/make_ensemble.py \
  runs/p0-kd288x10-9600-a075t3-s42/ema_0.998.pt \
  runs/p0-kd288x10-9600-a075t3-s17/ema_0.998.pt \
  runs/p0-kd320x8-9600-a075t3-s17/ema_0.998.pt \
  runs/p0-kd288x9-9600-a075t3-s17/ema_0.998.pt \
  runs/p0-sgdr2-do005-wd0070-192x9x6-s17-9600/ema_0.998.pt \
  -o runs/p0-ens-superteach/div_w05_t105.pt --weights 1.5 1.0 1.0 1.0 0.5 --temperature 1.05

# fresh student from the super-teacher (GPU, ~48 min)
python train.py --implementation student --config configs/student_288x10x9_backout.json \
  --teacher-checkpoint runs/p0-ens-superteach/div_w05_t105.pt --kd-alpha 0.75 --kd-temperature 3 \
  --device cuda --precision bf16 --seed 7 --steps 9600 --schedule sgdr --cycles 2 \
  --optimizer muon --lr 0.02 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kdst288x10-9600-a075t3-s7

# gentle continuation, twice (~25 min each): low-lr WSD from the previous EMA
python train.py --implementation student --config configs/student_288x10x9_backout.json \
  --teacher-checkpoint runs/p0-ens-superteach/div_w05_t105.pt --kd-alpha 0.75 --kd-temperature 3 \
  --init-checkpoint runs/p0-kdst288x10-9600-a075t3-s7/ema_0.998.pt \
  --device cuda --precision bf16 --seed 7 --steps 4800 --schedule wsd --wsd-decay-steps 2400 \
  --optimizer muon --lr 0.004 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kdst288x10wsd-4800-a075t3-s7
python train.py --implementation student --config configs/student_288x10x9_backout.json \
  --teacher-checkpoint runs/p0-ens-superteach/div_w05_t105.pt --kd-alpha 0.75 --kd-temperature 3 \
  --init-checkpoint runs/p0-kdst288x10wsd-4800-a075t3-s7/ema_0.998.pt \
  --device cuda --precision bf16 --seed 7 --steps 4800 --schedule wsd --wsd-decay-steps 2400 \
  --optimizer muon --lr 0.004 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kdst288x10wsd2-4800-a075t3-s7

# third submission's frozen evaluation (CPU, FP32)
python evaluate.py --checkpoint runs/p0-kdst288x10wsd2-4800-a075t3-s7/ema_0.998.pt --device cpu --precision fp32 --split test

# --- fourth round (final submission) ---
# teacher v3: 9-member ensemble, weights/temperature tuned on validation
python tools/make_ensemble.py \
  runs/p0-kd288x10-9600-a075t3-s42/ema_0.998.pt \
  runs/p0-kd288x10-9600-a075t3-s17/ema_0.998.pt \
  runs/p0-kd320x8-9600-a075t3-s17/ema_0.998.pt \
  runs/p0-kd288x9-9600-a075t3-s17/ema_0.998.pt \
  runs/p0-sgdr2-do005-wd0070-192x9x6-s17-9600/ema_0.998.pt \
  runs/p0-kdst288x10wsd2-4800-a075t3-s7/ema_0.998.pt \
  runs/p0-kdstm288x10wsd2-4800-a075t3-s7/ema_0.998.pt \
  runs/p0-plainm288x10-s17-9600/ema_0.998.pt \
  runs/p0-kdstm288x10-9600-a075t3-s42/ema_0.998.pt \
  -o runs/p0-ens-superteach/div9_t105.pt --weights 1.5 1 1 1 0.5 3 3 3 1.5 --temperature 1.05

# fresh student from teacher v3, modern architecture, extended training (GPU, ~2 h)
python train.py --implementation student --config configs/student_288x10x9_backout_modern.json \
  --teacher-checkpoint runs/p0-ens-superteach/div9_t105.pt --kd-alpha 0.75 --kd-temperature 3 \
  --device cuda --precision bf16 --seed 7 --steps 14400 --schedule sgdr --cycles 3 \
  --optimizer muon --lr 0.02 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kdstm288x10-14400-a075t3-s7-tv3

# gentle continuation, twice (~25 min each): low-lr WSD from the previous EMA
python train.py --implementation student --config configs/student_288x10x9_backout_modern.json \
  --teacher-checkpoint runs/p0-ens-superteach/div9_t105.pt --kd-alpha 0.75 --kd-temperature 3 \
  --init-checkpoint runs/p0-kdstm288x10-14400-a075t3-s7-tv3/ema_0.998.pt \
  --device cuda --precision bf16 --seed 7 --steps 4800 --schedule wsd --wsd-decay-steps 2400 \
  --optimizer muon --lr 0.004 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kdstm288x10wsd-4800-a075t3-s7-tv3b
python train.py --implementation student --config configs/student_288x10x9_backout_modern.json \
  --teacher-checkpoint runs/p0-ens-superteach/div9_t105.pt --kd-alpha 0.75 --kd-temperature 3 \
  --init-checkpoint runs/p0-kdstm288x10wsd-4800-a075t3-s7-tv3b/ema_0.999.pt \
  --device cuda --precision bf16 --seed 7 --steps 4800 --schedule wsd --wsd-decay-steps 2400 \
  --optimizer muon --lr 0.004 --aux-lr-scale 1 --weight-decay 0.07 --label-smoothing 0 \
  --dropout 0.05 --ema-decays 0.998 0.999 0.9995 --eval-every 1200 \
  --run-dir runs/p0-kdstm288x10wsd2-4800-a075t3-s7-tv3b

# final submission's frozen evaluation (CPU, FP32)
python evaluate.py --checkpoint runs/p0-kdstm288x10wsd2-4800-a075t3-s7-tv3b/ema_0.999.pt --device cpu --precision fp32 --split test
```
