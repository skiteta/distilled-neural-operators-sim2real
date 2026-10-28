# syouya_tobita: Track 1 method

## 1. Summary
A 4-block factorized neural operator with four interval heads (2.73 M parameters in total) is produced by a fixed 15-stage lineage:
- **S1–S7:** a 10-block hybrid spectral/convolutional operator is trained on the simulations and then on the 81 real cases. Its interval heads are trained with a smooth surrogate of the Safe Prediction Score (SPS).
- **S8–S12:** it is refined by self-distillation cycles with a differentiable copy of the composite score.
- **S13–S14:** simulation replay is added.
- **S15:** the model is distilled into the 4-block student.

At inference, a per-window turbulent-kinetic-energy calibration (TKECAL) rescales the predicted fluctuations to the turbulence level of the observed input. A CUDA-graph runtime keeps the Time score high.

What matters most for the score:
- the SPS-trained intervals (SPS carries weight .2);
- the composite-score-aware losses;
- the fast runtime (Time ≈ 93.4).

## 2. Model
**Architecture and number of parameters.**
- **S1–S4 (HybridFlowOperator), 3.42 M parameters.** 10 OperatorBlocks of width 128 plus a two-scale U-Net context (2 more OperatorBlocks at 1/2 and 1/4 resolution).
  - Each OperatorBlock is GroupNorm → (depthwise 5×5 conv + low-rank spectral mix, rank 32, 12×20 rfft2 modes + 1×1 mix), then GroupNorm → FFN (1×1 128→768, GLU, GELU, 1×1 384→128), with scaled residuals.
- **S5–S15 (FactorizedFlowOperator).** The same model plus an amplitude head and a direction-residual head that apply an energy-preserving rotation of the time-centred field. S5 initializes them so that the function is unchanged. The 10-block model has 3.73 M parameters.
- **Deployed model (S15).** 4 blocks copied from teacher blocks [0, 4, 5, 9], plus the multiscale context: 2,573,618 parameters.
- **Interval heads (151,640 parameters in total):**
  - a center head (FastIntervalHead on 186 flow-diagnostic channels of the observed input and the point prediction);
  - a radius head (two summed FastInterval networks);
  - a hidden-feature residual head on the backbone's final normalized feature map;
  - an observed-center residual head (hidden map plus the 40 time-centred observed u,v channels).

**Inputs and outputs.**
- Input: 20 frames of (u, v, p) on the 32×64 grid, in physical units. It is standardized per window (the scale is the per-window std of the observed u,v) and gets 2 coordinate channels and 12 descriptor channels.
- Output: the next 20 frames of (u, v), with pressure set to 0 as in the real data. The model predicts the last input frame, plus a learned trend, plus a 20-step correction.
- The interval is built as described in §5.

**Starting point.** Random initialization. No official baseline checkpoint and no external weights are used.

## 3. Data
**Data used and how it is combined.** Only the released `train_sim` (100 files, u, v, p) and `train_real` (81 files, u, v; `7575_0.h5` excluded).
- **S1:** pre-training on all 100 simulations, then fine-tuning on the real cases.
- **S2–S12:** real data only.
- **S13–S15:** each step adds 8 simulation windows to the 8 real windows, drawn from 99 simulations with the input pressure set to 0 (loss weight 0.3).
  - In S15, half of these windows come from the 18 simulated regimes that have no released real data.
- **Windows:** 20 input frames and 20 target frames, decimated by 2 to 32×64.
  - S1–S4 use the phase (0, 0) and starts every 20 frames.
  - S6–S15 use every stride-1 start (65,926 real windows).

**Validation split.** None for the submitted run. The five Reynolds-grouped folds of S1 are used only to produce out-of-fold predictions for training the first interval head; those fold models are discarded. Step counts and endpoints are fixed; nothing is selected on validation data inside `train()`.

**Augmentation (S8–S15).**
- a random 2×2 decimation phase;
- a temporal cadence of 2 with probability 0.25;
- a vertical reflection with probability 0.5 (the field is flipped and v is negated; losses are computed after mapping back).
- The teachers of S13–S15 give a reflection-averaged target ½[T(x) + R⁻¹T(Rx)].

**Data left out, and why.** `train_real/7575_0.h5`, a duplicate of another real case; the Decision-phase data omits it as well. The simulation `7575_0.h5` is also skipped in the S13–S15 replay (99 of 100 simulations).

## 4. Training
Common settings:
- Optimizer: AdamW (weight decay 1e-4). The point model and the heads are clipped separately at gradient norm 1.
- Precision: FP32, deterministic algorithms, cuBLAS workspace `:4096:8`, PyTorch defaults for TF32 (allowed for cuDNN convolutions, not for matmuls).
- Every stage runs as an unchanged historical trainer, driven by `runs/FINAL_REBUILD/driver/*`.
- Seeds are listed in the table and fixed in the code and plans.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps | Batch size |
|---|---|---|---|---|---|
| S1 sim | 100 simulations, 4,900 windows | L_point = relL2 + .25 TKE + .15 MVPE + .08 spectral-L1 + .02 divergence + .03 width-pinball | AdamW; lr 3e-4 → 1e-6, cosine; EMA .9995; seed 20260808 | 2,500 | 8 |
| S1 folds (×5) | real, Reynolds-grouped folds | L_point | lr 8e-5 → 1e-6, cosine | 8,000 each | 8 |
| S1 interval head | 876 out-of-fold windows | location + point term + pinball(q .88) − soft SPS reward (τ .04) + reg | lr 4e-4 → 1e-5, cosine | 2,000 | 2 |
| S1 joint | 81 real cases, 3,179 windows | L_point + 0.2 · interval loss | point lr 8e-5 → 1e-6, head lr 4e-4 → 1e-5, cosine; EMA .9995 | 8,000 | 8 |
| S2 center/radius heads | same, point frozen | −0.75 · soft SPS reward (τ .04 → .004) | lr 4e-4 → 1e-5, cosine | 4 × 2,000 | 8 |
| S3 hidden head | same, point frozen | −0.75 · soft SPS reward | lr 2e-4 → 2e-6, cosine | 4,000 | 8 |
| S4 observed-center head | same, point frozen | −0.75 · soft SPS reward (shift only) | lr 2e-4 → 2e-6, cosine | 2,000 | 8 |
| S5 | – | conversion to the factorized model (function-preserving, CPU) | – | 0 | – |
| S6 | all real stride-1 windows (65,926) | relL2 + TKE + .1 MVPE + .25 early3 − 6 softSPS + 2 phase + .2 late10 | point lr 1e-4, heads 3.2e-3; cosine to 1% over 18,464; seed 2026091727 | 9,232 | 8 |
| S7 | same (exact resume of S6) | same | same schedule | 4,616 more (to 13,848) | 8 |
| S8 | real + augmentation | (100 − C)/25, C = differentiable composite score (§8) with softSPS (τ .01) | point lr 1e-4, heads 8e-4; cosine over 36,928; EMA; seed 20260919799 | 18,464 | 8 |
| S9–S11 | same streams as S8 | S8 loss + 1.0 point KD + 0.1 hidden-feature KD (teacher = previous stage) | as S8 | 18,464 each | 8 |
| S12 | stream seed +100 | same | as S8 | 18,464 | 8 |
| S13, S14 | real + 8 simulation windows per step | S8 loss + KD to the reflection-averaged teacher + 0.3 sim relL2 | as S8; stream offsets +200 / +900 | 18,464 each | 8 real + 8 sim |
| S15 distillation | same, focus on regimes without real data | relL2 to the reflection-averaged teacher + 0.1 hidden KD; heads frozen | point lr 2e-4 | 9,232 | 8 + 8 |
| S15 fine-tune | same; stream offset +1800 | relL2 + TKE + .1 MVPE + .25 early3 − 6 softSPS(τ .01) + 2 phase + .2 late10, + point/hidden KD + interval KD + 0.3 sim relL2 | point lr 1e-4, heads 8e-4; cosine | 18,464 | 8 + 8 |

Loss terms:
- early3 / late10: relL2 on frames 0–2 / 10–19.
- phase: relL2 after rescaling the time-centred prediction to the energy of the target, per pixel.
- softSPS: a sigmoid relaxation of the official SPS on the full interval (weights .5/.3/.2 on relL2, TKE, MVPE).

Deployed weights:
- S6: the EMA point at step 9,232.
- S7: the S6 point plus the S7 EMA heads.
- S8–S15: the 50:50 average of the EMA checkpoints at the middle and the end of the fine-tune.

**How runs or checkpoints are selected.** Inside `train()`: not at all. Every stage has a fixed number of steps and a fixed endpoint, with no validation. The recipe itself was chosen during development from our own held-out evaluations (Reynolds and angle-of-attack folds of the training data) and from Codabench public-leaderboard feedback. In particular, the pre-C983 "old" real loss in S15 was chosen for the final Dev submission because of its public-leaderboard TKE.

**Seeds and run-to-run spread.**
- One run of each stage; the seeds are in the table and in the stage plans.
- With identical hardware and software the pipeline is bitwise deterministic: every stage of our full rebuild reproduced the weights behind submission 948431 bit for bit on an RTX A5500.
- On a different GPU the numerics differ, so the result is a new sample of the same recipe. We did not run multiple seeds of the full lineage, so its seed-to-seed spread is not measured.

## 5. Uncertainty bounds
`lower` and `upper` come from four small heads that see only the current window: the flow diagnostics of the observed input and the point prediction, and the backbone's feature map.
- shift = center-head shift + hidden-head Δshift + observed-center Δshift
- radius = radius-head radius × exp(hidden-head Δlog r)
- center = point + shift · scale
- lower/upper = center ∓ radius · scale, where scale is the per-window std of the observed u,v.

The heads are trained to maximize a smooth surrogate of the official SPS (S1–S4, then jointly in S6–S15). There is no conformal or post-hoc calibration step at runtime. TKECAL (§6) changes only `prediction`; the interval is built before it and is unchanged.

## 6. Inference
**What `predict` does per call.** `predict` is the Codabench `submission.py` API, loaded from `output_dir/runtime`.
- At import (untimed), the runtime:
  - sets a glibc `mallopt` heap policy (allocation only);
  - loads the weights on CUDA;
  - compiles two small kernels with NVRTC from the CUDA libraries shipped with torch.
- Per call:
  - windows are processed in groups of 50, as fixed batches of 5, padded by repeating the last window;
  - a CUDA graph of the forward pass is captured once on the first call and then replayed;
  - finiteness and `lower ≤ upper` are checked on the GPU;
  - results are copied to pinned host buffers.
- The output does not depend on how the input is split into calls (checked bitwise for N = 1, 7 and 12).

**Numeric precision.**
- FP32, except the two 1×1 FFN convolutions in the 4 main blocks, which run in FP16 (FP32 activations) through the NVRTC kernels.

**TKECAL.** Before returning, the point prediction p of each window is rescaled around its temporal mean m:
- p′ = m + r·(p − m), with r = clip((G∗k_obs / max(G∗k_pred, 1e-12))^0.05, 0.8, 1.25);
- k = ½ Σ over u,v of the temporal population variance of the 20 observed (k_obs) or predicted (k_pred) frames;
- G is a Gaussian blur with σ = 4 px and replicate padding.

It uses the same window only and runs inside the captured graph.

**Code identity.** The header docstring of `submission.py` predates the runtime changes and is kept byte-identical to the Codabench submission; this section describes the actual behaviour.

## 7. Speed
To keep the Time score high:
- the deployed student has only 4 blocks;
- inference runs as a replayed CUDA graph with a fixed batch of 5 (the official scored parts are 5 windows);
- a pinned, asynchronous device-to-host transfer is used;
- the FFN and depthwise kernels are fused (NVRTC);
- a glibc heap policy avoids page faults on the large result buffers.

Public Time score: 93.43 (948431).

## 8. What mattered
Public-leaderboard scores of our submissions:

| submission | composite | what changed |
|---|---|---|
| 941684 | 81.630 | a 4-block student distilled from the self-distilled 10-block model (the S1–S12 lineage) |
| 943857 | 81.738 (+0.11) | all-simulation replay during distillation; the fixed-batch-5 runtime; reflection test-time averaging (the last was not kept in the final lineage) |
| 945706 | 81.982 (+0.24) | two simulation-replay stages with a reflection-averaged teacher (S13–S14), distilled into the 4-block student (S15); the heap policy. The heap policy alone saved about 2.25 ms per window on the evaluation server (Time 93.41 at 3.63 ms/window). |
| 948431 | 82.033 (+0.05) | pre-C983 "old" real loss in S15 plus TKECAL. TKE +0.67; RelL2 −0.06 and MVPE −0.07; SPS +0.06. |

Our public score rose from 80.01 (2026-08-24) to 81.63 (941684) while the S1–S12 recipe was developed. We did not isolate the contribution of each of those stages on the public leaderboard.

**Where the fixed constants came from.** These are hyperparameters, not training data.
- **Composite loss (S8–S14).** The power-mean exponent (0.6854) and the weights (.5/.1/.1/.1/.2), with Time fixed at 90, were inferred from public leaderboard rows (subscores and final scores).
- **S1 interval constants.** Several were frozen using public-leaderboard history during development.
- **TKECAL.** It was chosen among 14 variants on our held-out predictions, and its exponent by a rule fixed before the confirming runs.
- No test labels or test inputs are used anywhere.

## 9. Compute
**Hardware.** 1 × NVIDIA RTX A5500 (24 GB), Intel Core i9-14900KF, 125 GB RAM.

**Training time.**
- **Submitted lineage:** the stages were trained on a shared GPU; the sum of the stage walls is 18.2 h.
- **Verified rebuild:** S11–S15 took 3.3 h unshared.
- **End-to-end `train()` in the submitted Docker image on this machine (16 CPU cores):** 21,605 s (6.0 h). Its weights and predictions are bit-identical to 948431.

**Peak GPU memory (PyTorch allocations).** 3.91 GiB in S13/S14; the trainers cap their allocator at 4 GiB (S1 nodes at 8 GiB). Peak host memory is 11.6 GB.

## 10. References
- Neural operators: Li et al., *Fourier Neural Operator for Parametric PDEs* (ICLR 2021). The spectral mixing in our blocks is a low-rank variant.
- RealPDEBench: Hu et al., ICLR 2026, and the RealPDE competition starting kit (the metric definitions follow its `scoring.py`).
- Knowledge distillation: Hinton et al., 2015. Mean-teacher/EMA weights: Tarvainen & Valpola, 2017.
- Library: PyTorch 2.2.2 (BSD-style license), with NumPy, h5py and PyYAML. No third-party model code is included; all model and training code is ours (MIT license).
