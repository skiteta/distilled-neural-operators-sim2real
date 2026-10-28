# Training pipeline

`package/solution.py:train()` builds the deployed model from scratch in 15 fixed stages. It needs no validation data and does no checkpoint selection: every stage has a fixed number of updates and a fixed endpoint. On one RTX A5500 the whole run takes about 6 hours, and PyTorch allocates at most 3.9 GiB of GPU memory.

The readable components in `dno/` cover the objectives and augmentation of these stages, and `tests/` checks them against the trainers.

## Overview

```
simulations (100) ──► S1 pre-train ─┐
real cases (81) ─────► S1 Reynolds folds ─► out-of-fold errors ─► S1 interval head
                                    └─► S1 joint fine-tune ─► S2-S4 SPS-trained interval heads
                                                                 │
S5 factorized decoder (function-preserving) ◄────────────────────┘
S6-S7  real-data training with the full objective (point + intervals)
S8     composite-score training with augmentation (2x2 phase, cadence, reflection)
S9-S12 self-distillation cycles (teacher = previous stage, 10 blocks -> 10 blocks)
S13-S14 + simulation replay, reflection-averaged teacher
S15    distillation into the 4-block student (+ interval distillation, focus on regimes without real data)
packaging: EMA averaging, runtime code, speed transforms, TKECAL
```

## Stages

| Stage | What it does | Objective (readable form in `dno/`) | Updates × batch | Trainer in `package/` |
|---|---|---|---|---|
| S1 | Pre-trains on the 100 simulations, then runs 5 Reynolds-grouped folds whose out-of-fold errors train the first interval head. A joint fine-tune on the 81 real cases follows. | RelL2 + 0.25 TKE + 0.15 MVPE + 0.08 spectral L1 + 0.02 divergence + 0.03 width pinball; interval head: location + pinball − soft SPS | 2,500 × 8 / 5 × 8,000 × 8 / 2,000 × 2 / 8,000 × 8 | `src/realpde/c628_pipeline.py`, `scripts/run_c252_release_only_c180_scratch_recipe.py` |
| S2 | Trains the center and radius heads; the point model is frozen. | −0.75 × `soft_sps_reward` (τ 0.04 → 0.004) | 4 × 2,000 × 8 | `src/realpde/fixed_sps_head_training.py` |
| S3 | Trains the hidden-feature head. | −0.75 × `soft_sps_reward` | 4,000 × 8 | `src/realpde/fixed_hidden_sps_training.py` |
| S4 | Trains the observed-center head. | −0.75 × `soft_sps_reward` (shift only) | 2,000 × 8 | `src/realpde/fixed_observed_center_training.py` |
| S5 | Converts the model to the factorized decoder (direction / amplitude / fixed-energy rotation); the function is unchanged. | — | 0 | `runs/FINAL_REBUILD/driver/stage5_c728p.py` |
| S6 | Trains on all 65,926 stride-1 real windows. | `final_stage_loss` form with a scheduled τ | 9,232 × 8 | `runs/C758R_full81_exposure_matched/train.py` |
| S7 | Resumes S6 exactly and continues the heads. | same | +4,616 × 8 | `runs/C787R_full81_point_preserved/train.py` |
| S8 | Composite-score training with augmentation (`augment.sample_window`, `augment.reflect`). | `composite_score_loss` with `soft_sps_reward` (τ 0.01) | 18,464 × 8 | `runs/LATCADF_full81/train.py` |
| S9–S11 | Self-distillation: the teacher is the previous stage, and the student is a copy with 10 blocks. | composite + `distillation_loss` | 18,464 × 8 each | `runs/SD10/train_sd10.py` |
| S12 | Same as S9–S11, with a new data stream. | same | 18,464 × 8 | `runs/CYCLE4/train_cycle4.py` |
| S13, S14 | Adds 8 simulation windows per step and uses a reflection-averaged teacher (`augment.reflection_averaged`). | composite + `distillation_loss` + 0.3 × simulation RelL2 | 18,464 × (8 + 8) each | `runs/TTAKD/train_simat.py`, `runs/SIMALL/sim_replay_all.py` |
| S15 | Builds a 4-block student from teacher blocks 0, 4, 5 and 9. It first distills (heads frozen), then fine-tunes. Half of the simulation windows come from the 18 regimes without real data. | distill: RelL2 to the teacher + 0.1 hidden KD. Fine-tune: `final_stage_loss` + `distillation_loss` + `interval_distillation_loss` + 0.3 × simulation RelL2. | 9,232 + 18,464 × (8 + 8) | `runs/TTAKD/train_simati_focus_old.py`, `runs/SIMALL/sim_replay_focus.py` |
| package | Averages the S15 checkpoints (50:50 EMA at the middle and the end of the fine-tune) and writes the runtime code. Applies the speed transforms (fixed batch 5, heap policy) and TKECAL. | — | — | `runs/FINAL_REBUILD/driver/package_13_15.py`, `tkecal_patch.py` |

Common settings:
- **Optimizer:** AdamW (weight decay 1e-4) with cosine schedules. The point model and the heads are clipped separately at gradient norm 1.
- **Numerics:** FP32 with deterministic algorithms.
- **Seeds:** fixed; see each stage's plan.

[method.md](method.md) gives the learning rates, seeds and the S1 node details.

`tests/test_training_components.py::test_stage_s15_objective_equals_trainer` runs the actual S15 objective, compiled from the trainer's own source. It checks that the result equals the composition written above.

## Why the package looks the way it does

The package was built to reproduce the submitted weights **bit for bit**, not to be read:
- **Stage trainers:** each stage runs the trainer that produced the submission, unchanged. The directory names under `package/runs/` are the internal names of those historical runs.
- **Drivers:** the drivers in `package/runs/FINAL_REBUILD/driver/` write each stage's plan, verify every code file by sha256, and replay the data streams on CPU before spending GPU time. After training they check the stream digests.
- **Library:** `package/src/realpde/` is pinned as a whole, so it also contains research modules that the final recipe does not use.

The result: `train()` inside the Docker image reproduced every stage's weights and the predictions of submission 948431 exactly on an RTX A5500 (33 of 33 recorded state hashes). On other GPUs the floating-point results differ, so a retrained model is a new sample of the same recipe.

## Internal run names in `package/runs/`

| Directory | Role |
|---|---|
| `FINAL_REBUILD/driver` | stage drivers, packaging, vendored runtime code |
| `C758R_full81_exposure_matched`, `C787R_full81_point_preserved` | S6, S7 trainers |
| `LATCADF_full81`, `LATCAD_upperRe` | S8 trainer and packaging helpers |
| `SD10`, `CYCLE4` | S9–S11 and S12 trainers, and the shared packaging function |
| `TTAKD`, `SIMALL` | S13–S15 trainers and the simulation-replay modules |
| `C1031R_depth_distillation` | reflection helper used by S9–S15 |
| `EXACTLANE` | the speed transforms of the runtime (fixed batch 5, heap policy) |
