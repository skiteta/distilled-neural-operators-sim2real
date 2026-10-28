"""Constants and stdlib-only helpers for the FINAL_REBUILD stage 8-12 driver (rebuild_8_12.py).

Nothing here imports torch; torch work runs in CPU-only or GPU child processes (same rule as common.py).
Code pins are ASSERTED. Weight / checkpoint / ZIP hashes of the historical lineage are RECORDED ONLY.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

from common import DRIVER, ROOT  # noqa: F401  (re-exported for the 8-12 modules)

# ---------------------------------------------------------------- training / helper code (asserted)
T8 = 'runs/LATCADF_full81/train.py'                         # stage 8 LATCADFH (lattice 2x2 + cadence-2 p=.25)
T8_REFLECTION = 'runs/LATCADF_full81/reflection_transform.py'   # imported by T8 from its own directory (train.py:33)
T_SD10 = 'runs/SD10/train_sd10.py'                          # stages 9-11 (self-distillation continuation)
T_CYCLE4 = 'runs/CYCLE4/train_cycle4.py'                    # stage 12 (and the LC5-style extra cycle)
T_SIM = 'runs/SIMALL/train_sim.py'                          # --sim-replay all (SIMA variant)
SIM_MODULE = 'runs/SIMALL/sim_replay_all.py'                # imported by T_SIM (train_sim.py:76)
SD10_REFLECTION = 'runs/C1031R_depth_distillation/reflection_transform.py'   # plan['reflection_source'] (code file)
OBJECTIVE = 'scripts/inferred_score_objective.py'
TRAIN_SCRIPTS = (T8, T_SD10, T_CYCLE4, T_SIM)

# Static import closure (import_closure below) of the four training scripts + the SD10 reflection source.
# Every entry is asserted at driver start and pinned again in each generated plan's source_pins.
CODE_PINS_8_12 = {
    'runs/CYCLE4/train_cycle4.py': '9ae899831fccc8ee87761f3af0ec943341b8febd2260419a3f26a6dc4a7320cd',
    'runs/LATCADF_full81/reflection_transform.py': '627f8b031068730bb69f705c30583e6c2c20b815b93b22e9c7f399041a4aa004',
    'runs/LATCADF_full81/train.py': '23e3e866e6fb7b19f511adb678a224e7105664f4b60f0999a3093f3ef6907e8a',
    'runs/SD10/train_sd10.py': '18adfa13199f77672db2f23f789c5673172140015f41c22c7b1e87c452c23f9c',
    'runs/SIMALL/sim_replay_all.py': 'a35fdb9ef7fb6508e4cac10204c5e4f9653a7047323022d62b191c8d795b03c7',
    'runs/SIMALL/train_sim.py': '31e5b32a5b4b75611199cefeb71f5832293b0da49d42b3a764112f331a5c6177',
    'runs/C1031R_depth_distillation/reflection_transform.py': '627f8b031068730bb69f705c30583e6c2c20b815b93b22e9c7f399041a4aa004',
    'scripts/experiment_records.py': 'b7ed0c45278c57e383a25cf7e2ec62fcb5bb1574535382217cb072ea21b3e66b',
    'scripts/inferred_score_objective.py': '1df375040345b24480c36e0ce3b64be77624798fb062258d565522544e0d64a0',
    'src/realpde/__init__.py': '0d12c507525545b048e8961c722249f977dc9ee817a05162758c89f6bb52d2e9',
    'src/realpde/data.py': '1291c43ef2db4f6e31f178d3f08e5e44649006fc265a97b4a40903581a97cb98',
    'src/realpde/fast_interval.py': 'd45406ab13f1c512b5a63045ba2e6fcecef6842f6a054d980fea86acf1f859a2',
    'src/realpde/fixed_sps_head_training.py': '047fe69ba0c862f941592f07c11f37588823e26ea93b4e2dce5c90341bab18ee',
    'src/realpde/flow_diagnostics.py': '828cdf6b08f7bf6c0285885b7ddfa6148f3d8c66284ecbf8217f765b0b9f116d',
    'src/realpde/hidden_sps_interval.py': '5f2f20105837097373444cd1eee1132870e8da6bc2fb87c134c66d65e898b3b4',
    'src/realpde/identity_interval_hardening.py': '28651e132b7f216971071c9385085d244e7468d7993ba67e7b1f92bc87f95b40',
    'src/realpde/losses.py': '4b0e80b41858385b79314ec1b6680102d575d5043c582cd4cc099d996f65345f',
    'src/realpde/metrics.py': 'ebd534c2ec6e6201abf7a0aace2b870b85452d8beebafbbfe1eb58eb1a9fad8d',
    'src/realpde/model.py': '2e30152f3eff53b945044e35937ddee69605d5faa02155be5f1a6fefbe02ac6c',
    'src/realpde/released_lattice_dataset.py': '266e1c44d752b5e4159e722b9fb445dab693227ee06a0191873638ad4a43e24f',
    'src/realpde/sps_radius_capacity.py': 'ed9b8ffbf67e36b635b68278c6e39b108ca697e2d46ba5ed7ec213410184c729',
    'src/realpde/temporal_start_dataset.py': '6db3969c245d281089de18863a5e837f7d2597ccf16c17192bcd06a12c3b178f',
    'src/realpde/train.py': 'ed4f8eeab783f22206489d610ec48b40a8ba5e93f74b9c64761761fe1761e9b9',
}
# Historical packaging code whose logic package_8_12.py lifts (recorded; never executed by the rebuild).
LIFTED_FROM = {
    'runs/LATCAD_upperRe/package_latcad.py': '94115be8bafe9cd93ddb63566d824cf8f518ca0d3d5503a401dea6a5790a249d',
    'runs/LATCADF_full81/finalize_latcadf.py': 'f30da141d196e8c5d0a45b085f3d6202f519a0c03fbc899f962fdf4cc13e6f5f',
    'runs/SD10/finalize_sd10.py': '51312c99b90403a4d9df9411b075522c7d1cce0be090263aedb210bb620d5c6c',
    'runs/CYCLE4/finish_generic.py': '06c23c5b9f4aa2bcbf70d0f023fcac690b6ede44d0bdff6bd773f25e2ae52ac6',
}

# ---------------------------------------------------------------- vendored runtime (C973H family, no weights)
# Non-weight members of runs/C973R_async_release/C973H_ASYNC_TRANSFER_RESEARCH.zip (the historical
# plan['runtime_baseline'] of stages 8-12), vendored once as code/text. The member ORDER is the ZIP order
# (weights included as slots); package_8_12 writes packages in this order exactly like package_latcad.py /
# finalize_sd10.py did. Weight bytes never come from here.
RUNTIME_C973_DIR = DRIVER / 'runtime_code_c973h'
RUNTIME_C973_ORDER = [
    'factorized_model.py', 'center_head.pt', 'observed_center_head.pt', 'LICENSE', 'model.pt', 'submission.py',
    'hidden_head.pt', 'radius_head.pt', 'realpde/model.py', 'realpde/fused_bias_ffn_cuda.py', 'realpde/fast_interval.py',
    'realpde/__init__.py', 'realpde/hidden_sps_interval.py', 'realpde/sps_radius_capacity.py', 'realpde/flow_diagnostics.py',
    'training_recipe/reflection_fraction.py.txt', 'training_recipe/average_state.py.txt',
    'training_recipe/source_train.py.txt', 'training_recipe/reflection_transform.py.txt', 'training_recipe/combination.json',
    'realpde/fused_half_ffn_cuda.py', 'realpde/fused_local_residual_cuda.py', 'training_recipe/fused_local_runtime.json',
    'training_recipe/deployment_loader_recovery.json', 'async_transfer.py']
RUNTIME_C973 = {
    'factorized_model.py': '58be092335223326b2ecacd64d489a7fc87f3582e23131525233e09e838126fc',
    'LICENSE': '0a6b51e48c63e6eb105b6aae1639bca755ee3d0b04cebab4b92d266527ebb22b',
    'submission.py': 'a1b8a94c4d66c7a7b37a5ec5e567ecccccdf70d794a55088cd19e8a8bd563e44',
    'realpde/model.py': '2e30152f3eff53b945044e35937ddee69605d5faa02155be5f1a6fefbe02ac6c',
    'realpde/fused_bias_ffn_cuda.py': '7fd3043f888fece7bdaf44ff957fbf03b5401c180369829281ceba724ba0eaeb',
    'realpde/fast_interval.py': 'd45406ab13f1c512b5a63045ba2e6fcecef6842f6a054d980fea86acf1f859a2',
    'realpde/__init__.py': '0d12c507525545b048e8961c722249f977dc9ee817a05162758c89f6bb52d2e9',
    'realpde/hidden_sps_interval.py': '5f2f20105837097373444cd1eee1132870e8da6bc2fb87c134c66d65e898b3b4',
    'realpde/sps_radius_capacity.py': 'ed9b8ffbf67e36b635b68278c6e39b108ca697e2d46ba5ed7ec213410184c729',
    'realpde/flow_diagnostics.py': '828cdf6b08f7bf6c0285885b7ddfa6148f3d8c66284ecbf8217f765b0b9f116d',
    'training_recipe/reflection_fraction.py.txt': 'cf62edb4bcee4edb857f6d0ae4c15bffd1c8aa9e4b930e2620f8947735240184',
    'training_recipe/average_state.py.txt': 'ed409a518d1488e9c601a6240987d738972bf11d3c56127a196299a4e6638ebd',
    'training_recipe/source_train.py.txt': 'e4e812f8cbf044456389379542219feb350563eea9ae41265e1ec8136239212c',
    'training_recipe/reflection_transform.py.txt': '627f8b031068730bb69f705c30583e6c2c20b815b93b22e9c7f399041a4aa004',
    'training_recipe/combination.json': '58c0fc41e8bee5f533c6e7bb247fc393617749f67f9b40011bc478a26674afe5',
    'realpde/fused_half_ffn_cuda.py': 'def6ed1e8d248330374c48fa17a1d14636ad1411191cdab58633bf8cad8d8929',
    'realpde/fused_local_residual_cuda.py': 'c5ef2e123c92ec02e3a829b89c1846594f4daed260102c42509723ebd186a80d',
    'training_recipe/fused_local_runtime.json': '944eb1be8de886ec4c0df5aa730958fc0066d8f3feda03bd159d23eff5da22e5',
    'training_recipe/deployment_loader_recovery.json': 'b4664bf0838a24be11c1adb2f41e9e177e56d28fbfefc90701f030f62d6b3da5',
    'async_transfer.py': '6ad8ce419d348d6a65231e22c61fe774320244cdfa801b54e8b9205cc7f68ea0',
}
RUNTIME_C973_SOURCE = {'zip': 'runs/C973R_async_release/C973H_ASYNC_TRANSFER_RESEARCH.zip',
                       'zip_sha256': '58e1cb3006331a892fb25802b9fd53298e60f9a0e695aaa8d0e6f3cfd6a1b3b2',
                       'note': 'vendored 2026-09-24 (non-weight members only); ZIP recorded, never read by the driver'}
WEIGHTS = {'model.pt': 'point', 'center_head.pt': 'center', 'radius_head.pt': 'radius', 'hidden_head.pt': 'hidden',
           'observed_center_head.pt': 'observed'}         # finalize_sd10.py:7 / package_latcad.py:5 (same order)

# ---------------------------------------------------------------- recipe policy (inlined constants)
SEED = 20260919799               # recipe seed of stages 8-12 (LATCADF_full81 .. LC4B_full81 plans)
HELD_FOLD_SEEDS = {'upperRe': 20260919796, 'AoA15': 20260919797, 'AoA20': 20260919798}   # historical held surfaces (record)
REFERENCE_WINDOWS, REFERENCE_STEPS = 42847, 6000   # C757R exposure rule, identical to rebuild_1_7 (common.py)
SAMPLING_META = {'candidate': 'C728D', 'stride': 1, 'support_frames': 40,
                 'interpretation': 'Qualified dense native windows with trailing support40'}


def base_steps(windows):
    """S6 step count: ceil(6000*windows/42847) (C758R prepare.py:10; 9232 for the release, 6000/7259/7457 held)."""
    return math.ceil(REFERENCE_STEPS * windows / REFERENCE_WINDOWS)


def ema(base):
    return .999 ** (6000 / base)


# runs/LATCADF_full81/candidate_plan.json 'training', verbatim except the base-derived numbers.
def latcadf_training(base, seed):
    return {
        'steps': 2 * base, 'schedule_steps': 4 * base, 'seed': seed, 'batch': 8, 'point_lr': 0.0001, 'head_lr': 0.0008,
        'EMA': ema(base), 'loss': 'Unchanged empirical composite; no teacher or auxiliary loss',
        'tau': '.01 fixed; operative value',
        'optimizer': 'Same fresh AdamW pointLR1e-4/headLR8e-4, seed, full cosine exposure, EMA and sample stream.',
        'sampling': 'complete shuffled epochs, final partial batch retained',
        'endpoint': 'Fixed midpoint/finalEMA50:50; no evaluation-dependent selection.', 'wave_weight': 0.0,
        'late_weight': 0.0,
        'phase_formula': 'pc*(norm(tc,time+uv)/clamp(norm(pc,time+uv),1e-8)); supervised training loss only, no target at runtime.',
        'sps_weight': None, 'tau_fixed': 0.01, 'phase_energy_normalized': True, 'point_trainable': True,
        'observed_width': False, 'reflection_active': True, 'objective_family': 'empirical_composite_fixed_time90',
        'objective_source': OBJECTIVE, 'objective_sha256': CODE_PINS_8_12[OBJECTIVE],
        'diagnostic_only_losses': ['energy_normalized_phase', 'early3', 'late10'], 'backbone_dropout_probability': 0.03,
        'backbone_dropout_modules': 12, 'training_mode': 'Point train,4heads eval; EMA eval; deterministic deployed inference',
        'new_updates': 2 * base, 'lattice_phases': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'lattice_selection': 'Uniform phase per sampled original window, separate CPU generator(seed+3)',
        'lattice_phase_seed': seed + 3, 'cadence2_probability': 0.25}


# runs/LCSD_full81/plan.json 'training' (= SD10_L_full81 + cadence 0.25), verbatim except the base numbers;
# LC4B adds stream_seed_offset/replay_teacher_stream, SIMA adds sim_replay (key order as in those plans).
def sd10_training(base, seed, offset=None, replay=True, sim_replay=None):
    t = {'seed': seed, 'steps': 2 * base, 'warmup_steps': 0, 'finetune_steps': 2 * base,
         'retained_teacher_blocks': list(range(10)), 'teacher_depth': 10, 'student_depth': 10, 'point_lr': 0.0001,
         'head_lr': 0.0008, 'EMA': ema(base), 'dropout': 0.03, 'observed_width': False, 'objective_source': OBJECTIVE,
         'objective_sha256': CODE_PINS_8_12[OBJECTIVE],
         'loss': 'Warm: relativeL2(student,teacher)+.1 normalized hidden MSE. Fine: empirical real composite +same distillation. Teacher always eval/frozen; no teacher in inference.',
         'cadence2_probability': 0.25, 'lattice_phases': [0, 1, 2, 3]}
    if offset is not None:
        t.update(stream_seed_offset=offset, replay_teacher_stream=replay)
    if sim_replay is not None:
        t['sim_replay'] = sim_replay
    return t


# Stage table. 'final' marks the continuation cycle that --sim-replay all converts into the SIMA variant.
STAGES = {
    's8': {'candidate': 'LATCADFH', 'kind': 'latcadf', 'script': T8,
           'note': 'LATCADF: init = stage-7 runtime (C787H-equivalent), no teacher; lattice 2x2 phases + cadence spacing-2 p=0.25 + backbone dropout 0.03 + composite loss'},
    's9': {'candidate': 'LCSDH', 'kind': 'sd10', 'script': T_SD10, 'offset': None, 'replay': True,
           'note': 'full-depth self-distillation continuation of LATCADFH (lattice+cadence full81 fit)'},
    's10': {'candidate': 'LCSD2H', 'kind': 'sd10', 'script': T_SD10, 'offset': None, 'replay': True,
            'note': 'second full-depth continuation cycle (teacher LCSDH)'},
    's11': {'candidate': 'LCSD3H', 'kind': 'sd10', 'script': T_SD10, 'offset': None, 'replay': True,
            'note': 'third full-depth continuation cycle (teacher LCSD2H)'},
    's12': {'candidate': 'LC4BH', 'kind': 'sd10', 'script': T_CYCLE4, 'offset': 100, 'replay': False,
            'note': 'fourth continuation cycle with KD and a fresh data order (stream seed offset 100)'},
    's13': {'candidate': 'LC5BH', 'kind': 'sd10', 'script': T_CYCLE4, 'offset': 200, 'replay': False,
            'note': 'LC5-style extra continuation cycle: KD from stage 12 and another fresh data order (offset 200, runs/CYCLE5/rule.json LC5B)'},
}
SIM_VARIANT = {'s12': {'candidate': 'LC4BSIMH', 'note': 'stage 12 as the SIMA variant: LC4B cycle (offset 100) + 0.1 x relL2 on all released simulations (train_sim.py, sim_replay=all); NO historical counterpart'},
               's13': {'candidate': 'SIMAH', 'note': 'SIMA: fresh-order continuation cycle of LC4B (real stream identical to LC5B, offset 200) + 0.1 x relL2 on ALL released simulations (C1019 replay loss, point parameters only)'}}

# ---------------------------------------------------------------- historical lineage (read-only; smoke stand-ins + records)
HIST_C787H_ZIP = 'runs/C787R_full81_point_preserved/C787H_EMA13848_RESEARCH.zip'
HIST_C758_PLAN = 'runs/C758R_full81_exposure_matched/candidate_plan.json'
HIST_C1017H_ZIP = 'runs/C1017R_backbone_dropout_composite/C1017H_DROPOUT_COMPOSITE_RESEARCH.zip'   # historical s8 payload template (B2)
HIST = {
    's8': {'dir': 'runs/LATCADF_full81', 'plan': 'runs/LATCADF_full81/candidate_plan.json',
           'zip': 'runs/LATCADF_full81/LATCADFH_CANDIDATE_RELEASED_LATTICE_CADENCE_RESEARCH.zip',
           'checkpoint': 'runs/LATCADF_full81/candidate_deployed_checkpoint.pt'},
    's9': {'dir': 'runs/LCSD_full81', 'plan': 'runs/LCSD_full81/plan.json', 'zip': 'runs/LCSD_full81/LCSDH_SD10_RESEARCH.zip',
           'checkpoint': 'runs/LCSD_full81/LCSDH_deployed_checkpoint.pt'},
    's10': {'dir': 'runs/LCSD2_full81', 'plan': 'runs/LCSD2_full81/plan.json', 'zip': 'runs/LCSD2_full81/LCSD2H_SD10_RESEARCH.zip',
            # LCSD3H was packaged from this earlier ZIP (duplicate recipe members) via teacher_path_override (B6)
            'zip_used_as_teacher': 'runs/LCSD2_full81/INVALID_duplicate_members/LCSD2H_SD10_RESEARCH.zip',
            'checkpoint': 'runs/LCSD2_full81/LCSD2H_deployed_checkpoint.pt'},
    's11': {'dir': 'runs/LCSD3_full81', 'plan': 'runs/LCSD3_full81/plan.json', 'zip': 'runs/LCSD3_full81/LCSD3H_SD10_RESEARCH.zip',
            'checkpoint': 'runs/LCSD3_full81/LCSD3H_deployed_checkpoint.pt'},
    's12': {'dir': 'runs/LC4B_full81', 'plan': 'runs/LC4B_full81/plan.json', 'zip': 'runs/LC4B_full81/LC4BH_SD10_RESEARCH.zip',
            'submission_zip': 'submissions/20260924_131335_JST__READY__LC4BH_LATCAD_SD4.zip',
            'checkpoint': 'runs/LC4B_full81/LC4BH_deployed_checkpoint.pt'},
    's13_sim': {'dir': 'runs/SIMA_full81', 'plan': 'runs/SIMA_full81/plan.json', 'zip': 'runs/SIMA_full81/SIMAH_SD10_RESEARCH.zip',
                'checkpoint': 'runs/SIMA_full81/SIMAH_deployed_checkpoint.pt'},
}
HIST_HELD = {   # historical held surfaces (fold smoke comparison only); names by driver stage
    'upperRe': {'s8': 'runs/LATCAD_upperRe/candidate_plan.json', 's9': 'runs/LCSD_upperRe/plan.json',
                's10': 'runs/LCSD2_upperRe/plan.json', 's11': 'runs/LCSD3_upperRe/plan.json', 's12': 'runs/LC4B_upperRe/plan.json',
                's13': 'runs/LC5B_upperRe/plan.json', 's13_sim': 'runs/SIMA_upperRe/plan.json'},
    'AoA15': {'s8': 'runs/LATCAD_AoA15/candidate_plan.json', 's9': 'runs/LCSD_AoA15/plan.json', 's10': 'runs/LCSD2_AoA15/plan.json',
              's11': 'runs/LCSD3_AoA15/plan.json', 's12': 'runs/LC4B_AoA15/plan.json', 's13': 'runs/LC5B_AoA15/plan.json',
              's13_sim': 'runs/SIMA_AoA15/plan.json'},
    'AoA20': {'s8': 'runs/LATCAD_AoA20/candidate_plan.json', 's9': 'runs/LCSD_AoA20/plan.json', 's10': 'runs/LCSD2_AoA20/plan.json',
              's11': 'runs/LCSD3_AoA20/plan.json', 's12': 'runs/LC4B_AoA20/plan.json', 's13': 'runs/LC5B_AoA20/plan.json',
              's13_sim': 'runs/SIMA_AoA20/plan.json'},
}
# Recorded reference values (full release). Stream digests depend only on seed + case list, so a full-release
# rebuild is expected to MATCH them; weight states will differ (recorded only, like common.HISTORICAL).
HISTORICAL_8_12 = {
    's8': {'batch_digest': '62cc0d371b5bee5d679b97b89fd70c04c9fb6678a871efaefc11cfd377e24f70',
           'augmentation_digest': '02239695229849ffdedc04f719817b04cf2eee83f682f52fc96437ec335d7e6b',
           'phase_digest': 'f606470438bb3b07638afc93da6f0f7c93d90f2c6f2726d93aa2d5d5042625d1',
           'cadence_digest': '2ea6b83589d0d623bdc0710e850b3698af688da4e95b3c885d07f78ec696a59b',
           'middle_batch_digest': '8b28f987dd50fc7f0008defa34b18735764c6c7dd87689131d0da05c7b149f60',
           'phase_counts': [37268, 36445, 37059, 36936], 'cadence_counts': [112730, 34978], 'cadence_trailing_fallbacks': 1751,
           'initial_states': {'point': '32e98c8c7505c21a54292925904a8cc7b2c20c6061a1e258ab757a285e99ca0f',
                              'center': 'f785456b3d24ecff141ad45d7779c0925ff28401f3f22aa6d57236e260a170f7',
                              'radius': '734997b07152b4fd3deee84692aee5c44351592209f9aa10fcc02c79249b05df',
                              'hidden': 'a0b6c05e3af1680c8c9102eccb2afebd57e2105538a449f91d6b2e23fe18c116',
                              'observed': '992a1fe16a4c5aec186e8ea9a7074de02a6e8dd03ffd4dcdd7b76bac7df54da5'},
           'zip_sha256': '75c7373dba81cdf37175eee2a251bd88112b069d3b1a3292d9daa0d819c60b20',
           'deployed_point': '7848f5688fda67bb26e5c0005a4e2adbbaf1bb0d6f2accad0e48f6bbe7ac3e1d',
           'training_loop_seconds': 9244.3},
    's9': {'batch_digest': '62cc0d371b5bee5d679b97b89fd70c04c9fb6678a871efaefc11cfd377e24f70',
           'augmentation_digest': '02239695229849ffdedc04f719817b04cf2eee83f682f52fc96437ec335d7e6b',
           'lattice_phase_counts': [37268, 36445, 37059, 36936], 'cadence_counts': [112730, 34978],
           'cadence_trailing_fallbacks': 1751, 'zip_sha256': '4b24d1c707c329a694172691989003c4d7f7cd7056efe5c614986d2fe97e6ac1',
           'deployed_point': '314e014403dd84da873cfa35114e44f37570588302be7f9ab24181b54df6ec46', 'training_loop_seconds': 5833.6},
    's10': {'batch_digest': '62cc0d371b5bee5d679b97b89fd70c04c9fb6678a871efaefc11cfd377e24f70',
            'augmentation_digest': '02239695229849ffdedc04f719817b04cf2eee83f682f52fc96437ec335d7e6b',
            'lattice_phase_counts': [37268, 36445, 37059, 36936], 'cadence_counts': [112730, 34978],
            'cadence_trailing_fallbacks': 1751, 'zip_sha256': '809348354e0c9a4fb4ff396343307887512d788dc3556212dbcc65e55c863d2e',
            'deployed_point': '109277e5db90d454c5a3963628069ed537725fade330092b4164d04e82046a4b', 'training_loop_seconds': 5609.2},
    's11': {'batch_digest': '62cc0d371b5bee5d679b97b89fd70c04c9fb6678a871efaefc11cfd377e24f70',
            'augmentation_digest': '02239695229849ffdedc04f719817b04cf2eee83f682f52fc96437ec335d7e6b',
            'lattice_phase_counts': [37268, 36445, 37059, 36936], 'cadence_counts': [112730, 34978],
            'cadence_trailing_fallbacks': 1751, 'zip_sha256': '727fd138cf36d06ded9f07db93211e99da6733782dbf4373f5be6903f87b6009',
            'deployed_point': '2c5940e82eb911d9725231c2fda7dc58da820b19f35e8c036e9c634f9bdcd074', 'training_loop_seconds': 1893.0},
    's12': {'batch_digest': '1441c7c2f80c5cc3c9a6d1cca7bf67edc646ef96b4514df564803d803c65b4d4',
            'augmentation_digest': '2e200d99a69887f459db40c6214261145752f77a55be6b3c51f54cd210bc59e5',
            'lattice_phase_counts': [37145, 36961, 36742, 36860], 'cadence_counts': [112731, 34977],
            'cadence_trailing_fallbacks': 1814, 'zip_sha256': '3c4ce397e7344378f34f947d68f7132e298785125d6c31fc6323a741c6667055',
            'deployed_point': '2e43a25f00eb174f08a5210167ba7ca2225659ef3ab22aba0eb85d31f02f4e63', 'training_loop_seconds': 11064.2},
    's13': {'batch_digest': 'e41595029352b5541ea8c937d9ced0b9bbad7665ccd6a7a494330b0241898ef1',   # offset 200 (SIMA real stream = LC5B)
            'augmentation_digest': 'e4b10e3875fdd5a19b4ebd7d97ae1fb3fcaba5f99ef8aff881db1e2d4bf1ccd1',
            'lattice_phase_counts': [37105, 36922, 36677, 37004], 'cadence_counts': [112658, 35050],
            'cadence_trailing_fallbacks': 1795},
    's13_sim': {'sim_digest': '3fe81944c168c4a18b9f1c04d472f1af08fb865106048742592cf3616afb8ffb', 'sim_samples': 147707,
                'sim_windows': 95139, 'zip_sha256': 'ebc0de188bca78346fbb1e3715ec17d0a74f597a768846922ba7f8c02211ef9e',
                'deployed_point': '828d01e37f086478a66ff763600a31adabdb240bac9194a08361a833a6adbd53', 'training_loop_seconds': 11268.2},
}

# ---------------------------------------------------------------- GPU estimates (one RTX A5500), from historical logs
# Source: training_seconds of training_candidate/curve.jsonl (the rows printed to train.log) and report.json
# budget.wall_seconds. 'unshared' = per-step cost when the GPU was not shared x steps + start-up:
#   LATCADF 0.0844 s/step (fastest logged 100-step segments; the full81 run was mostly shared),
#   SD10/CYCLE4 0.1001 s/step (LCSD3H ran alone: 1,893 s wall for 18,464 steps incl. ~38 s start-up),
#   SIM 0.1416 s/step (fastest SIMA_full81 segment; one extra point forward/backward on 8 simulation windows).
# 'logged' = historical full81 wall with the A5500 shared by other jobs (2-7x slower). Folds scale with steps.
GPU_ESTIMATES_8_12 = {
    'latcadf': {'per_step': 0.0844, 'overhead': 45, 'logged': {'s8': 9287},
                'note': 'LATCADF_full81: loop 9,244 s / wall 9,287 s shared; device peak 0.73 GB (4 GiB allocator cap)'},
    'sd10': {'per_step': 0.1001, 'overhead': 45, 'logged': {'s9': 5834, 's10': 5609, 's11': 1893, 's12': 11065},
             'note': 'LCSD3H 1,893 s alone; LCSDH 5,834 s, LCSD2H 5,609 s, LC4BH 11,065 s shared; device peak 2.3 GB; '
                     'LC5B (s13) ran on held surfaces only'},
    'sim': {'per_step': 0.1416, 'overhead': 100, 'logged': {'s13': 11269},
            'note': 'SIMA_full81 11,269 s shared; device peak 3.9 GiB of the 4 GiB allocator cap (train_sim.py sets it)'},
    'package': {'note': 'CPU ~3-5 s per stage: two 65 MB checkpoints, five torch.save payloads, deterministic ZIP'},
}


def estimate_seconds(kind, steps, key=None):
    """[unshared estimate, historical logged wall scaled to `steps` (None if never run at full81)]."""
    e = GPU_ESTIMATES_8_12[kind]
    logged = e['logged'].get(key)
    return [round(e['per_step'] * steps + e['overhead']), None if logged is None else round(logged * steps / 18464)]


# ---------------------------------------------------------------- static helpers
def _module_files(name):
    parts = name.split('.')
    if parts[0] != 'realpde':
        return []
    out = [ROOT / 'src/realpde/__init__.py']
    base = ROOT / 'src' / Path(*parts)
    for cand in (base.with_suffix('.py'), base / '__init__.py'):
        if cand.is_file():
            out.append(cand)
            break
    return out


def import_closure(entries, search_dirs):
    """First-party static import closure: realpde.* under src/, relative imports, and top-level modules that
    resolve to a file in search_dirs (the script's own directory and scripts/, as the scripts set sys.path)."""
    todo, seen = [(ROOT / e).resolve() for e in entries], set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            found = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found += _module_files(alias.name) or [Path(d) / (alias.name.replace('.', '/') + '.py') for d in search_dirs]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    mod = base / Path(*node.module.split('.')) if node.module else base
                    found += [mod.with_suffix('.py'), mod / '__init__.py'] + [mod / (a.name + '.py') for a in node.names]
                else:
                    found += _module_files(node.module) or [Path(d) / (node.module.replace('.', '/') + '.py') for d in search_dirs]
                    for a in node.names:
                        found += _module_files(node.module + '.' + a.name)
            todo += [f.resolve() for f in found if f.is_file() and f.resolve() not in seen]
    return sorted(str(p.relative_to(ROOT)) for p in seen)


def script_closure(script):
    return import_closure([script], [(ROOT / script).parent, ROOT / 'scripts'])


def plan_fields_read(script, names=('m', 'parent', 'manifest')):
    """Plan paths a script reads (m['a']['b'], m['a'].get('c')), from its AST; used by the smoke plan comparison."""
    out = set()
    for node in ast.walk(ast.parse((ROOT / script).read_text())):
        keys, n = [], node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get' \
                and node.args and isinstance(node.args[0], ast.Constant):
            keys, n = [node.args[0].value], node.func.value
        elif not isinstance(node, ast.Subscript):
            continue
        while isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
            keys.append(n.slice.value)
            n = n.value
        if isinstance(n, ast.Name) and n.id in names and keys and all(isinstance(k, str) for k in keys):
            out.add((n.id,) + tuple(reversed(keys)))
    # keep only the longest paths (m.training is implied by m.training.seed)
    return sorted(p for p in out if not any(q != p and q[:len(p)] == p for q in out))
