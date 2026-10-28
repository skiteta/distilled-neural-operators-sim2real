# FINAL_REBUILD stages 13-15 + final packaging (`rebuild_13_15.py`)

Continues a COMPLETE `rebuild_8_12.py` output (stage 12 = LC4BH-equivalent) and rebuilds the rest of the lineage of the
READY candidate `submissions/20260927_082107_JST__READY__SIMSAT2D4FHF5MA_FIXED5.zip` (audit:
`runs/FINAL_REPRO_AUDIT/lineage_audit_20260927.md`, stages S13-S15 + packaging P):

| stage | candidate | trainer (run unchanged, sha256 asserted) | teacher / start | recipe |
|---|---|---|---|---|
| s13 | SIMSATH | `runs/TTAKD/train_simat.py` + `runs/SIMALL/sim_replay_all.py` | stage 12 (full-depth copy) | fine-tune 2*S6 = 18,464, reflection-averaged point KD (`teacher_tta`), all 99 released simulations at weight .3, stream offset 200, no teacher-stream replay |
| s14 | SIMSAT2H | same | s13 | same, offset 900 |
| s15 | SIMSAT2D4FH | `runs/TTAKD/train_simati_focus.py` + `runs/SIMALL/sim_replay_focus.py` | s14, 4-block student from teacher blocks [0,4,5,9] | distillation S6 = 9,232 + fine-tune 2*S6 = 18,464, interval KD (centre 1.0, log-width .1), focus fraction .5 (4 of 8 simulation windows from the 18 regimes without trained real data), offset 1800 |
| final | SIMSAT2D4FHF5MA | packaging only (CPU) | s15 package | finalize_sd10 `package()` (4-block `depth_runtime`) -> `make_fixed5.py` (F5) -> `make_malloc.py` (MA) |

S6 = ceil(6000 * windows / 42847) = 9,232 on the release; EMA .999**(6000/S6); middle checkpoint = distillation + fine-tune/2;
deployed weights = midpoint/final fine-tune EMA 50:50 (as `finalize_sd10.py`). Recipe seed and holdout fold are inherited
from the stage-12 run (full release: seed 20260919799).

S15 options: `--loss-mode old` = `runs/TTAKD/train_simati_focus_old.py` (pre-C983 real loss; historical E11 plan
`runs/SIMSAT2D4FO_full81/plan.json`, candidate SIMSAT2D4FOH); `--focus-fraction F` (per-batch focus count round(8F);
candidate SIMSAT2D4F[O]K<k>H); `--focus-ratio R` = the focus2 sampler (`train_simati_focus2[_old].py` +
`sim_replay_focus2.py`, fixed per-window emphasis, accumulator; candidate SIMSAT2D4M[O]R<R>[S<offset>]H; used by the held
E8/E11 runs and by the E9 run `runs/SIMSAT2D4X_full81` = `--focus-ratio 9 --s15-offset 2100`, which this driver never reads);
`--s15-offset N` (real-data stream offset, default 1800). Steps, teacher and student surgery are the same for every S15 variant.

## New files (nothing existing was modified)
`runs/FINAL_REBUILD/driver/`
- `rebuild_13_15.py` - the driver (subclass of the stage 8-12 `Driver`: same logging, child-process runs, seals, plan
  writing, holdout records, `--stop-before-gpu` gate, report); plans, CPU checks, GPU launch, post-training checks,
  packaging, final ZIP, smoke comparisons.
- `common_13_15.py` - code pins (asserted), recipe builders, historical references (recorded), GPU estimates, AST helpers.
- `rebind_13_15.py` - runs a trainer file unchanged as `__main__` (`python SCRIPT --run DIR [--cpu-preflight]`
  semantics); folds: loads the trainer's simulation module under its own name and rebinds `discover_cases` in memory to
  drop the held regimes (stage-1-7 fold rule, as `rebind_8_12.py`).
- `preflight_13_15.py` - CPU tools: `replay` (every seeded stream, both phases, checkpoint digests, all/focus/focus2
  simulation streams), `states` (the trainers' model set-up on CPU -> teacher/initial state hashes), `runtime-input`.
- `package_13_15.py` - `stage` (finalize_sd10 `package()` for any depth, reusing `package_8_12.py` helpers) and
  `final` (F5 + MA member transforms, deterministic ZIPs, member checks of `make_fixed5.py`/`make_malloc.py`/`stage_ready_d4.py`).
- `runtime_smoke_13_15.py` - CPU import + output-contract smoke of a packaged runtime (fresh interpreter, CUDA hidden).
- `README_13_15.md` - this file.

`runs/FINAL_REBUILD/smoke_13_15/` (verification only): `run_suite.sh`, `rebuild_mode_test/make_stage12_standin.py`
(MOCK stage-12 dir from copies of `runs/LC4B_full81`), `rebuild_mode_test/mock_training.py` (copies of the historical
S13-S15 training outputs as mocked GPU results), and the suite outputs (`*.console.log`, per-run `driver.log`,
`rebuild_report.json`, `verification/`).

## How to run (project root = the repository root; any cwd works, paths are relative to the root)
Prerequisite: a COMPLETE stage-12 directory from `rebuild_8_12.py` run WITHOUT `--sim-replay all` (its s12 must be
LC4BH; `--extra-cycle` does not matter). From the existing verified stage-7 rebuild this is (GPU, ~2.6 h unshared):
```
.venv310/bin/python runs/FINAL_REBUILD/driver/rebuild_8_12.py --stage7 runs/FINAL_REBUILD/R1_full --out runs/FINAL_REBUILD/R1_full_8_12
```
Stages 13-15 + deliverable (GPU, ~2.7 h unshared on one A5500):
```
# optional first: all plans + CPU checks of s13, stop before the first GPU launch (CPU only)
.venv310/bin/python runs/FINAL_REBUILD/driver/rebuild_13_15.py --stage12 runs/FINAL_REBUILD/R1_full_8_12 --out runs/FINAL_REBUILD/R1_full_13_15 --stop-before-gpu
# full run (use --resume if the directory already exists, e.g. after the command above or an interruption)
.venv310/bin/python runs/FINAL_REBUILD/driver/rebuild_13_15.py --stage12 runs/FINAL_REBUILD/R1_full_8_12 --out runs/FINAL_REBUILD/R1_full_13_15 --resume
# variants (own --out): --loss-mode old | --focus-fraction F | --focus-ratio R | --s15-offset N
```
Deliverable: `<out>/final_simsat2d4fhf5ma/SIMSAT2D4FHF5MA_FIXED5_REBUILT.zip` (its members extracted in `runtime/`,
audit `final_audit.json`, CPU runtime smoke `runtime_smoke.json`); report `<out>/rebuild_report.json`, log `<out>/driver.log`,
per-command logs `<out>/logs/`. A GPU launch is exactly the command stored in `rebuild_report.json`
(`stages.sN.gpu_command` after `--stop-before-gpu`), with the environment of `common.ENV` plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (as the historical launches).
`--resume` requires the same options, re-verifies every seal (sha256, incl. all seals of the stage-12 directory), reuses
sealed stages only if their plan equals the plan regenerated from the current inputs, and refuses stale CPU-check files;
a crashed GPU stage (a `training_candidate/` without `report.json`) must be moved aside by hand and restarts from step 0
(the trainers have no mid-run resume).

CPU verification (no GPU): `bash runs/FINAL_REBUILD/smoke_13_15/run_suite.sh` (every `--out` must not exist), or a single
`rebuild_13_15.py --smoke --out runs/FINAL_REBUILD/smoke_13_15/<name> [--stage12 DIR] [--holdout-re ...|--holdout-aoa ...] [--seed N] [S15 options]`.

## Per stage
plan (generated, never copied) -> CPU checks -> GPU training -> post-training checks -> package -> seal
- plan: pins only code (static import closure of the trainer + its simulation module + `reflection_source`) and the
  previous rebuilt stage's `runtime/` members; `runtime` = previous `runtime/`, `teacher_source` = previous stage dir,
  `teacher_archive` = previous rebuilt package (payload templates + carried `training_recipe/`), `runtime_baseline` =
  vendored `runtime_code_c973h/`. No historical weights/ZIPs/checkpoints, no absolute paths.
- CPU checks: code pins; the trainer's own `--cpu-preflight` (one window, both phases, no update; PASS required);
  `preflight_13_15.py states` (teacher hash, student surgery, initial point/heads hashes); `preflight_13_15.py replay`
  (every stream); the teacher checkpoint `teacher_source/training_candidate/resume_<fine>.pt`, which the trainers load only
  after their last step (train_simat.py:207), must exist and be sealed before any GPU time is spent.
- GPU launch: a probe in the training environment must see CUDA and the historical launch gate of free device memory
  (4,800 MiB for s13/s14, 3,500 MiB for s15), otherwise the driver stops before the trainer starts.
- after training: `report.json` COMPLETE with the planned steps; every phase digest/count and the simulation digest,
  samples, unique windows, calls equal the CPU replay; `job.json` teacher/initial state hashes and kept blocks equal the
  CPU prediction.
- package (`package_13_15.py stage`) and seal (plan, training_script.txt, report/job, all resume_*.pt, package audit,
  ZIP, deployed checkpoint, runtime files). Final: F5+MA ZIP, code members must equal the READY ZIP code members
  (asserted: deterministic transforms of pinned code), CPU runtime smoke must PASS, sealed.
- historical comparison (recorded, not asserted in rebuild mode): streams (expected equal on the full release), initial
  states and deployed weight states (equal only if every earlier stage were bit-exact).

## Verification performed (2026-09-27, CPU only; `runs/FINAL_REBUILD/smoke_13_15/`)
Full release with the historical lineage as read-only stand-ins (`full_default`):
- plans: s13/s14/s15 vs `runs/SIMSAT_full81`, `runs/SIMSAT2_full81`, `runs/SIMSAT2D4F_full81` plans: all fields any
  trainer/simulation module/packager reads equal except the intended ones (source_pins, runtime, runtime_baseline B2);
  whole `training` and `sampling` dicts equal.
- streams: 15/15, 15/15, 23/23 historical values reproduced by the CPU replay (phase batch/augmentation digests,
  lattice/cadence counts, trailing fallbacks, checkpoint digests at 9232/18464, simulation digest/samples/unique
  windows/first sample ids; s15 distillation + fine-tune + focus stream).
- initial states: CPU set-up on the historical teacher runtimes reproduces the historical `job.json` teacher and initial
  state hashes of all three stages (incl. the 4-block surgery, 0b2ef221...).
- trainer `--cpu-preflight`: PASS for all three generated plans.
- packaging: `package_13_15.py stage` on the historical training outputs reproduces SIMSATH 36/36, SIMSAT2H 39/39,
  SIMSAT2D4FH 42/42 ZIP members and all three deployed checkpoints byte-identically (historical plan); with the rebuilt
  plan only `<CAND>_continuation.json` differs (intended). Final: F5 43/43 and F5MA 43/43 members byte-identical (same
  order) to `runs/EXACTLANE/SIMSAT2D4FHF5*/...` = the READY ZIP; CPU runtime smoke PASS for the rebuilt final runtime
  and the READY ZIP runtime with identical CPU prediction hashes (depth 4, 6 operator blocks, batch 5, split invariant).
- variants: `--loss-mode old` (plan = S15 + the recorded E11 changes, 23/23 streams, preflight of
  train_simati_focus_old.py PASS), `--focus-fraction 0.25` (real streams 17/17), `--focus-ratio 4.5 --loss-mode old`
  (focus2 trainer PASS; on the release ratio 4.5 gives exactly fraction .5, and its replayed simulation stream equals the
  historical S15 focus stream digest), `--focus-ratio 9 --s15-offset 2100` (training dict identical to the E9 derivation
  of `runs/TTAKD/chain_e9.sh` applied to `runs/SIMSAT2D4_full81/plan.json`; E9's own directory is not read).
- folds: upperRe/AoA15/AoA20 with the held seeds: plans vs the held-surface plans OK, s13/s14 real streams 7/7 equal to
  the held reports, the held runs' all-simulation protocol replays their simulation digests exactly (the fold-rule stream
  differs by design); the held S15 runs were stopped before completion (no report to compare). A list fold
  (Re 8850,10125 + AoA15) runs through.
- rebuild mode on a MOCK stage-12 dir with mocked GPU steps (copies of the historical S13-S15 training outputs):
  stop-before-GPU, resume, per-stage post-training checks, packaging, seals, final ZIP, runtime smoke -> COMPLETE; the
  resulting deliverable equals the READY ZIP in 40/43 members (all 5 weights, all code, order), differing only in the
  three `*_continuation.json` recipe members (rebuilt plans); full reuse after completion (plans regenerated and equal); a
  tampered sealed file is detected; `--resume` with other options is refused; a launch without visible CUDA stops in the
  pre-launch probe (no trainer process started).
See `runs/FINAL_REBUILD/smoke_13_15/run_suite.console.log` for the suite summary.

## Remaining dependencies (explicit)
Code read or executed by a rebuild (all sha256-pinned in `common_13_15.py`, asserted at start and in every plan):
1. Trainers and simulation modules in experiment directories: `runs/TTAKD/train_simat.py`,
   `runs/TTAKD/train_simati_focus.py` (+ `_old`, `focus2`, `focus2_old` for the variants), `runs/SIMALL/sim_replay_all.py`,
   `runs/SIMALL/sim_replay_focus.py` (+ `sim_replay_focus2.py`). `train_simati_focus_old.py`/`train_simati_focus2_old.py`
   were created on 2026-09-27 and are used by running jobs; they are pinned at their current bytes (a later edit makes the
   driver refuse to start).
2. Shared closure: `src/realpde/` (14 files), `scripts/experiment_records.py`, `scripts/inferred_score_objective.py`
   (its loss constants were fitted to the public-leaderboard composite: provenance note, audit item 9).
3. `runs/C1031R_depth_distillation/reflection_transform.py` (plan `reflection_source`): a code file inside a non-lineage
   run directory, kept at the historical path as in stages 9-12; byte-identical to `runs/LATCADF_full81/reflection_transform.py`.
4. Packaging: `runtime_code_c973h/` (vendored C973H runtime code, incl. seven C973H-family `training_recipe/` texts that
   every package carries, as historically); `scripts/depth_distillation_cycle_v2.py` (only the function `depth_runtime` is
   compiled from the pinned file; the module is never imported because it imports unrelated scripts);
   `runs/EXACTLANE/make_fixed5.py` and `runs/EXACTLANE/make_malloc.py` (only their literal strings are read by AST;
   the scripts are not executed: they hard-code the repository path, run full81 scoring, qualification and timing and
   write into runs/EXACTLANE and runs/SOUP_20260923); `runs/EXACTLANE/fixed_transfer.py` (shipped as a ZIP member).
5. Stage 8-12 driver modules reused unchanged: `common.py`, `common_8_12.py`, `rebuild_8_12.py`, `rebuild_1_7.py`,
   `c675_ext.py`, `preflight_8_12.py`, `package_8_12.py`, `rebind_8_12.py`.
Data: `data/extracted/train_real` (81 cases; 7575_0.h5 excluded) and `data/extracted/train_sim` (100 files; the
simulation modules exclude 7575_0.h5 -> 99 replayed; folds also drop the held regimes); h5 metadata of all files is read.
Input: the stage-12 directory (every seal in it is re-verified; the stage 8-12 driver documents its own dependencies).
Environment: `.venv310` (Python 3.10.16, torch 2.2.2+cu121, numpy 1.26.4, h5py 3.11.0; recorded in `environment.json`),
one CUDA GPU; `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic algorithms), `CUDA_MODULE_LOADING=LAZY`, 4 BLAS threads,
`PYTHONDONTWRITEBYTECODE=1`; GPU runs add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Historical artifacts: never read by a rebuild. `--smoke` reads (read-only) `runs/LC4B_full81`, `runs/SIMSAT_full81`,
`runs/SIMSAT2_full81`, `runs/SIMSAT2D4F_full81`, `runs/EXACTLANE/SIMSAT2D4FHF5/`, `runs/EXACTLANE/SIMSAT2D4FHF5MA/`, the READY
ZIP and, for fold smokes, `runs/{LC4B,SIMSAT,SIMSAT2,SIMSAT2D4F}_{upperRe,AoA15,AoA20}`. `runs/SIMSAT2D4FO_full81` is never
read (its plan differences were recorded once as constants).
Removed relative to the historical chain: plan pins of C1017H ZIP/checkpoint/resume, C1031R teacher_runtime, C1017R
plan/confirmation, C728R evaluation contract and qualification scripts; the C973H weight-bearing ZIP (vendored code); teacher
runtimes copied/extracted by inline chain-script Python (now the previous stage's sealed `runtime/`); `finish_generic.py`'s
hard-coded root and evaluation-only reads (`runs/LATCADF_full81/full81_cases.json`, SOUP evaluators, `runs/C1099R_released_lattice/plan.json`,
qualification, timing); the teacher-ZIP hash assertion now refers to the previous rebuilt package.

## Risks / not verified
- No GPU was used: the three GPU trainings, the packaging of freshly trained outputs and the final ZIP built from them
  are untested; stage 12 has never been GPU-rebuilt either (the stage 8-12 driver is CPU-smoke-tested only). The mock
  chain exercises every driver step with historical training outputs in place of the GPU runs.
- Bitwise weight reproducibility is not expected across GPUs/driver versions; the driver guarantees identical code,
  plans, data streams and start states. The rebuilt weights (and predictions/score) will differ from the READY ZIP and must
  be evaluated anew; historical scores do not transfer.
- The deliverable is not qualified or timed by the driver (C894 qualification, C973 timing, official-image checks, harness
  replica are evaluation/selection steps). Run them on the rebuilt ZIP before any use. The CPU runtime smoke only checks
  import, the 4-block assertion, output contract, determinism and split invariance on CPU.
- Memory guards: the trainers cap the CUDA allocator at 4 GiB (`set_per_process_memory_fraction`) and host RSS at 20 GiB;
  S13/S14 historically peaked at 3.91 GiB device (with expandable segments), S15 at 2.3 GB, host ~8.6 GB. A shared or
  different GPU/allocator can OOM; historically such runs were relaunched. A crash restarts that stage from step 0.
- `training_recipe/` of the deliverable is carried as in the historical chain (plus the stage-8-12 B5 correction); it does
  not document S1-S7 (audit item 5); the three continuation JSONs embed the rebuilt plans.
- `runs/SD10/finalize_sd10.py` changed on 2026-09-26 16:55 UTC (EMA-mix/no-multiscale options): S15 was packaged with the
  current version, S13/S14 with the earlier one; both give identical results for this lineage (lift verified byte-for-byte above).
- `--loss-mode old` has no finished historical full81 run to compare packages with (E11 is still training); its plan,
  streams, start states and preflight are verified. The focus-ratio full81 run (E9, `runs/SIMSAT2D4X_full81`) is not read;
  only its recipe derivation (`chain_e9.sh`) is compared.

## Not done (outside this CPU-only task)
- The GPU runs themselves: stage 8-12 (`rebuild_8_12.py` from `runs/FINAL_REBUILD/R1_full`) and then stages 13-15 here.
- Qualification/timing and a quality evaluation of the rebuilt deliverable (the historical gates of `finish_generic.py`,
  `make_fixed5.py`, `make_malloc.py`, `stage_d4.sh`), and any decision to submit it.
- Rewriting `training_recipe/` into a complete S1-S15 record (audit suggestion A); the driver keeps the historical
  member layout so that the package stays comparable with the READY ZIP.

## GPU time (one RTX A5500)
| stage | updates | unshared estimate | historical logged wall (shared GPU) |
|---|---|---|---|
| s13 SIMSATH | 18,464 | ~3.2k s (modelled: 0.170 s/step + 100 s; +-25 %) | 17,206 s (4 concurrent jobs) |
| s14 SIMSAT2H | 18,464 | ~3.2k s (same model) | 18,260 s (4 concurrent jobs) |
| s15 SIMSAT2D4FH (any variant) | 9,232 + 18,464 | ~3.2k s (measured: SIMSAT2D4_full81, identical compute, ran alone: 3,155 s) | 22,852 s |
| packaging + final | - | CPU, < 1 min | - |
| total s13-s15 | | ~9.6k s (~2.7 GPU-h) | 58,318 s (~16.2 GPU-h) |
Model for s13/s14 (no uncontended SIMSAT-type log exists): LCSD3H alone 0.0851 s/step (10-block student fwd+bwd + teacher
fwd) and SIMSAT2D4_full81 alone (0.068 / 0.1136 s/step) give teacher fwd ~0.014 s and 10-block student fwd+bwd ~0.071 s;
a SIMSAT step = 2 student fwd+bwd (real + simulation) + 2 teacher fwd (TTA) ~0.170 s. Stage 8-12: ~9.2k s unshared (its README).
