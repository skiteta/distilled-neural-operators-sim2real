"""Constants and stdlib-only helpers for the FINAL_REBUILD stage 13-15 driver (rebuild_13_15.py).

Nothing here imports torch (same rule as common.py / common_8_12.py). Code pins are ASSERTED. Historical weights,
checkpoints and ZIPs are RECORDED ONLY; historical stream digests depend only on seed + case lists, so a full-release
rebuild with the recipe seed is expected to MATCH them (the smoke asserts that).
"""
from __future__ import annotations

import ast
import hashlib

from common import DRIVER, ROOT  # noqa: F401  (re-exported)
from common_8_12 import CODE_PINS_8_12, OBJECTIVE, SD10_REFLECTION, SEED, import_closure, sd10_training  # noqa: F401

# ---------------------------------------------------------------- training code (asserted)
T_SIMAT = 'runs/TTAKD/train_simat.py'                       # S13 SIMSATH, S14 SIMSAT2H
T_FOCUS = 'runs/TTAKD/train_simati_focus.py'                # S15 SIMSAT2D4FH (C983 composite loss)
T_FOCUS_OLD = 'runs/TTAKD/train_simati_focus_old.py'        # S15 --loss-mode old (pre-C983 loss; E11 SIMSAT2D4FOH)
T_FOCUS2 = 'runs/TTAKD/train_simati_focus2.py'              # S15 --focus-ratio (fixed per-window emphasis; held only)
T_FOCUS2_OLD = 'runs/TTAKD/train_simati_focus2_old.py'      # S15 --focus-ratio --loss-mode old (held only)
SIM_ALL = 'runs/SIMALL/sim_replay_all.py'                   # imported by train_simat.py:79
SIM_FOCUS = 'runs/SIMALL/sim_replay_focus.py'               # imported by train_simati_focus*.py:84
SIM_FOCUS2 = 'runs/SIMALL/sim_replay_focus2.py'             # imported by train_simati_focus2*.py:84
SIM_OF = {T_SIMAT: SIM_ALL, T_FOCUS: SIM_FOCUS, T_FOCUS_OLD: SIM_FOCUS, T_FOCUS2: SIM_FOCUS2, T_FOCUS2_OLD: SIM_FOCUS2}
SIM_KIND = {SIM_ALL: 'all', SIM_FOCUS: 'focus', SIM_FOCUS2: 'focus2'}

# Shared part of the static import closure (identical files and hashes as common_8_12.CODE_PINS_8_12).
SHARED_CLOSURE = [
    'scripts/experiment_records.py', 'scripts/inferred_score_objective.py', 'src/realpde/__init__.py',
    'src/realpde/data.py', 'src/realpde/fast_interval.py', 'src/realpde/fixed_sps_head_training.py',
    'src/realpde/flow_diagnostics.py', 'src/realpde/hidden_sps_interval.py', 'src/realpde/identity_interval_hardening.py',
    'src/realpde/losses.py', 'src/realpde/metrics.py', 'src/realpde/model.py', 'src/realpde/released_lattice_dataset.py',
    'src/realpde/sps_radius_capacity.py', 'src/realpde/temporal_start_dataset.py', 'src/realpde/train.py',
    SIM_ALL, SD10_REFLECTION]
NEW_CODE_PINS = {
    T_SIMAT: '66307e52958797b59be86feb4fc9150258ca3e4e4bb63cee27cc73f56767be8a',
    T_FOCUS: '7f84ae83e4fde9be6b64e72ce18c49321afdbcc8a4fb4c4d2de5204cf365d5e4',
    T_FOCUS_OLD: 'ca7bec7bd611e85860411ffc49255abb5b0e7c626e6689e61ff4aaef7c84604f',
    T_FOCUS2: 'a1aac217983d816db0bcff3fde98e1e79a92828bc6195cc7dd2735d753b04794',
    T_FOCUS2_OLD: '42cd2452dd60e82ed6dc8645d535f57e6a4866ca40b416e9682f1ba0c4db1239',
    SIM_FOCUS: '72b8ee361eb62f269955f0df610abd30843d7636274078f941f1dc33cc872057',
    SIM_FOCUS2: 'f92d111d7c8226f2d4af90f29b5fc84ce7b3c3db1bb6fd5a0b62e3b9b0d62bfa',
}
CODE_PINS_13_15 = {**{f: CODE_PINS_8_12[f] for f in SHARED_CLOSURE}, **NEW_CODE_PINS}

# Packaging code that is EXECUTED or whose literal constants are USED (asserted):
DEPTH_SOURCE = 'scripts/depth_distillation_cycle_v2.py'   # depth_runtime() (AST-lifted function only; module not imported)
F5_SOURCE = 'runs/EXACTLANE/make_fixed5.py'               # old/new/tail + the two batch-size replacement pairs (AST)
MA_SOURCE = 'runs/EXACTLANE/make_malloc.py'               # BLOCK/ARENA/anchor + the M_TOP_PAD insertion point (AST)
FIXED_TRANSFER = 'runs/EXACTLANE/fixed_transfer.py'       # shipped as the ZIP member fixed_transfer.py (F5)
PACKAGING_PINS = {
    DEPTH_SOURCE: 'c13134aa0085f7fe85277cedb9e40daf607f88e80762f97891d8c0341d486821',
    F5_SOURCE: 'c7b9070127e47bfbe4065fb6cb775852296121ca95cdca9043e02eb6987d67d6',
    MA_SOURCE: '584054b2504c39d86f516e8b264de5e8a70b062232606163a01f038e741a93a0',
    FIXED_TRANSFER: '0bc1e80150be577d969f0efbf3e164132fc84389db84a6fc3c0b94a3c7933fd2',
}
# Historical packaging / hand-off code whose logic is lifted (recorded; never executed by the rebuild).
LIFTED_FROM_13_15 = {
    'runs/SD10/finalize_sd10.py': 'eb154625974aac578cd939ea1b4b2c2e03f99ea84801e0279948d8b67a4e7160',   # package() (S15 was packaged
    # with this version, 2026-09-26 16:55 UTC; S13/S14 with the earlier 51312c99.. recorded in common_8_12.LIFTED_FROM; same
    # result at final_weight .5 without no_multiscale)
    'runs/CYCLE4/finish_generic.py': '06c23c5b9f4aa2bcbf70d0f023fcac690b6ede44d0bdff6bd773f25e2ae52ac6',
    'runs/TTAKD/stage_ready_d4.py': '7359602c1aa2b11f28d153fd5972f26ecd5418ee5d98c8aac34e4365c1598193',   # READY member checks
    'runs/TTAKD/chain_e6.sh': '43320172209d9766d185bbf015c64cbeb980e29120ef9a17f6327fe8d63130d2',        # S15 plan derivation
    'runs/TTAKD/chain_e6c.sh': '2af121c680f3b09403718efce1c9cca59a9dbe45da81e4d80167077961f68360',       # packaging order
}

# ---------------------------------------------------------------- recipe (inlined; plans are generated, never copied)
SIM_WEIGHT = 0.3
OLD_LOSS_TEXT = ('Warm: relativeL2(student,teacher)+.1 normalized hidden MSE. Fine: pre-C983 real loss (RelL2+TKE+.1MVPE+'
                 '.25early3-6softSPS+2energyNormalizedPhase+.2late10) +same distillation +simulation replay loss unchanged. '
                 'Teacher always eval/frozen; no teacher in inference.')   # runs/SIMSAT2D4FO_full81/plan.json training.loss
D4_BLOCKS = [0, 4, 5, 9]
DEFAULT_FOCUS_FRACTION = 0.5
DEFAULT_S15_OFFSET = 1800


def simsat_training(base, seed, offset):
    """runs/SIMSAT_full81 / runs/SIMSAT2_full81 plan 'training' (key order kept): the SD10/CYCLE4 recipe with stream
    offset, no teacher-stream replay, simulation replay 'all' at weight .3 and the reflection-averaged KD teacher."""
    t = sd10_training(base, seed, offset, False, 'all')
    t.update(sim_weight=SIM_WEIGHT, teacher_tta=True)
    return t


def d4_training(base, seed, loss_mode='default', focus=('fraction', DEFAULT_FOCUS_FRACTION), offset=DEFAULT_S15_OFFSET):
    """runs/SIMSAT2D4F_full81 plan 'training', derived exactly as the chain scripts did: SIMSAT2 plan ->
    chain_simsat2d4.sh (4-block student, blocks [0,4,5,9], warm-up S6 + fine-tune 2*S6, interval KD, offset 1200) ->
    chain_e6.sh (sim_focus_fraction, offset 1800); --loss-mode old adds the E11 fields (runs/SIMSAT2D4FO_full81);
    focus ratio / other offsets follow the E8/E9 derivations (chain_e9.sh: sim_focus_ratio, stream_seed_offset)."""
    t = simsat_training(base, seed, 900)
    t.update(student_depth=4, retained_teacher_blocks=list(D4_BLOCKS), warmup_steps=base, finetune_steps=2 * base,
             steps=3 * base, stream_seed_offset=1200, interval_kd=True, interval_kd_center=1.0, interval_kd_radius=0.1,
             teacher_tta=True, sim_weight=SIM_WEIGHT)
    kind, value = focus
    if kind == 'fraction':
        t.update(sim_focus_fraction=float(value), stream_seed_offset=int(offset))
    else:
        t.update(sim_focus_ratio=float(value), stream_seed_offset=int(offset))
    if loss_mode == 'old':
        t['loss'] = OLD_LOSS_TEXT
        t['loss_mode'] = 'old'
    return t


def focus_per_batch(fraction):
    """sim_replay_focus.py:43 (Python round, as the module)."""
    return max(1, min(7, round(8 * float(fraction))))


def s15_variant(loss_mode, focus, offset=DEFAULT_S15_OFFSET):
    """(candidate, trainer, historical key or None) of stage 15 for the chosen options."""
    kind, value = focus
    old = loss_mode == 'old'
    stream = '' if int(offset) == DEFAULT_S15_OFFSET else f'S{int(offset)}'
    if kind == 'fraction':
        script = T_FOCUS_OLD if old else T_FOCUS
        k = focus_per_batch(value)
        name = 'SIMSAT2D4F' + ('O' if old else '') + ('' if float(value) == DEFAULT_FOCUS_FRACTION else f'K{k}') + stream + 'H'
        hist = ('s15_old' if old else 's15') if (float(value) == DEFAULT_FOCUS_FRACTION and not stream) else None
    else:
        script = T_FOCUS2_OLD if old else T_FOCUS2
        tag = ('%g' % float(value)).replace('.', 'p')
        name = 'SIMSAT2D4M' + ('O' if old else '') + 'R' + tag + stream + 'H'
        hist = None
    return name, script, hist


STAGES_13_15 = {
    's13': {'candidate': 'SIMSATH', 'script': T_SIMAT, 'offset': 200, 'depth': 10, 'kind': 'simsat', 'hist': 's13',
            'note': 'SIMSAT: KD from stage 12 (LC4BH) with the reflection-averaged teacher point target (teacher_tta), full-depth '
                    'student copy, fresh real stream (offset 200), replay of all released simulations at weight .3'},
    's14': {'candidate': 'SIMSAT2H', 'script': T_SIMAT, 'offset': 900, 'depth': 10, 'kind': 'simsat', 'hist': 's14',
            'note': 'SIMSAT2: one more reflection-averaged-teacher cycle from stage 13 (SIMSATH), simulation .3, real stream 900'},
    's15': {'candidate': 'SIMSAT2D4FH', 'script': T_FOCUS, 'offset': 1800, 'depth': 4, 'kind': 'd4f', 'hist': 's15',
            'note': '4-block student of stage 14 (SIMSAT2H; teacher blocks [0,4,5,9]): distillation S6 + fine-tune 2*S6, reflection-'
                    'averaged point and interval KD, focused simulation replay (half of every simulation batch from the regimes '
                    'without trained real data), real stream 1800'},
}

# ---------------------------------------------------------------- historical lineage (read-only; smoke stand-ins + records)
HIST_13_15 = {
    's12': {'dir': 'runs/LC4B_full81', 'plan': 'runs/LC4B_full81/plan.json', 'zip': 'runs/LC4B_full81/LC4BH_SD10_RESEARCH.zip',
            'checkpoint': 'runs/LC4B_full81/LC4BH_deployed_checkpoint.pt', 'candidate': 'LC4BH'},
    's13': {'dir': 'runs/SIMSAT_full81', 'plan': 'runs/SIMSAT_full81/plan.json', 'zip': 'runs/SIMSAT_full81/SIMSATH_SD10_RESEARCH.zip',
            'checkpoint': 'runs/SIMSAT_full81/SIMSATH_deployed_checkpoint.pt', 'candidate': 'SIMSATH'},
    's14': {'dir': 'runs/SIMSAT2_full81', 'plan': 'runs/SIMSAT2_full81/plan.json', 'zip': 'runs/SIMSAT2_full81/SIMSAT2H_SD10_RESEARCH.zip',
            'checkpoint': 'runs/SIMSAT2_full81/SIMSAT2H_deployed_checkpoint.pt', 'candidate': 'SIMSAT2H'},
    's15': {'dir': 'runs/SIMSAT2D4F_full81', 'plan': 'runs/SIMSAT2D4F_full81/plan.json',
            'zip': 'runs/SIMSAT2D4F_full81/SIMSAT2D4FH_SD10_RESEARCH.zip',
            'checkpoint': 'runs/SIMSAT2D4F_full81/SIMSAT2D4FH_deployed_checkpoint.pt', 'candidate': 'SIMSAT2D4FH'},
}
# E11 (runs/SIMSAT2D4FO_full81, in training on 2026-09-27; never read by this driver): its plan differs from the S15 plan
# only in candidate, runtime, one descriptive key, training.loss (OLD_LOSS_TEXT) and training.loss_mode='old' (read once,
# 2026-09-27, plan sha256 recorded). Same seed/offset/steps -> same real and simulation streams as S15.
HIST_S15_OLD = {'plan_path_recorded': 'runs/SIMSAT2D4FO_full81/plan.json',
                'plan_sha256_recorded': '732cfc1fa00cf94d98a2620e2b339cfd34f69016aae242c92cfdb51aa5975a72',
                'candidate': 'SIMSAT2D4FOH', 'training_changes': {'loss': OLD_LOSS_TEXT, 'loss_mode': 'old'},
                'training_script': [T_FOCUS_OLD, SIM_FOCUS]}
HIST_FINAL = {
    'f5': {'path': 'runs/EXACTLANE/SIMSAT2D4FHF5/SIMSAT2D4FHF5_FIXED5_RESEARCH.zip',
           'sha256': 'dadd7737c4087017236abb34768845c8b316d9dc11e3b6e1b7c188fd33b9bbcd'},
    'f5ma': {'path': 'runs/EXACTLANE/SIMSAT2D4FHF5MA/SIMSAT2D4FHF5MA_FIXED5_RESEARCH.zip',
             'sha256': 'afcc27c66dc050a4cbb1b79017a52803180805ef03012d84471c10aa793d389c'},
    'ready': {'path': 'submissions/20260927_082107_JST__READY__SIMSAT2D4FHF5MA_FIXED5.zip',
              'sha256': 'afcc27c66dc050a4cbb1b79017a52803180805ef03012d84471c10aa793d389c',
              'note': 'byte copy of the F5MA ZIP (stage_ready_d4.py:71 shutil.copyfile)'},
}
# Code members of the READY ZIP (4-block F5MA runtime). A rebuilt 4-block final ZIP must carry exactly these bytes
# (deterministic transforms of the vendored runtime code); weights and training_recipe/ are rebuilt/recorded only.
READY_CODE_MEMBERS = {
    'factorized_model.py': '58be092335223326b2ecacd64d489a7fc87f3582e23131525233e09e838126fc',
    'LICENSE': '0a6b51e48c63e6eb105b6aae1639bca755ee3d0b04cebab4b92d266527ebb22b',
    'submission.py': '8201b2cb36c4fcc459960a361dd078f4614d5911d046cafc55ccda0645dacd02',
    'realpde/model.py': '2e30152f3eff53b945044e35937ddee69605d5faa02155be5f1a6fefbe02ac6c',
    'realpde/fused_bias_ffn_cuda.py': '7fd3043f888fece7bdaf44ff957fbf03b5401c180369829281ceba724ba0eaeb',
    'realpde/fast_interval.py': 'd45406ab13f1c512b5a63045ba2e6fcecef6842f6a054d980fea86acf1f859a2',
    'realpde/__init__.py': '0d12c507525545b048e8961c722249f977dc9ee817a05162758c89f6bb52d2e9',
    'realpde/hidden_sps_interval.py': '5f2f20105837097373444cd1eee1132870e8da6bc2fb87c134c66d65e898b3b4',
    'realpde/sps_radius_capacity.py': 'ed9b8ffbf67e36b635b68278c6e39b108ca697e2d46ba5ed7ec213410184c729',
    'realpde/flow_diagnostics.py': '828cdf6b08f7bf6c0285885b7ddfa6148f3d8c66284ecbf8217f765b0b9f116d',
    'realpde/fused_half_ffn_cuda.py': 'def6ed1e8d248330374c48fa17a1d14636ad1411191cdab58633bf8cad8d8929',
    'realpde/fused_local_residual_cuda.py': 'c5ef2e123c92ec02e3a829b89c1846594f4daed260102c42509723ebd186a80d',
    'async_transfer.py': '6ad8ce419d348d6a65231e22c61fe774320244cdfa801b54e8b9205cc7f68ea0',
    'fixed_transfer.py': '0bc1e80150be577d969f0efbf3e164132fc84389db84a6fc3c0b94a3c7933fd2',
}
READY_MEMBER_ORDER = [
    'factorized_model.py', 'center_head.pt', 'observed_center_head.pt', 'LICENSE', 'model.pt', 'submission.py', 'hidden_head.pt',
    'radius_head.pt', 'realpde/model.py', 'realpde/fused_bias_ffn_cuda.py', 'realpde/fast_interval.py', 'realpde/__init__.py',
    'realpde/hidden_sps_interval.py', 'realpde/sps_radius_capacity.py', 'realpde/flow_diagnostics.py',
    'realpde/fused_half_ffn_cuda.py', 'realpde/fused_local_residual_cuda.py', 'async_transfer.py']   # then training_recipe/*, fixed_transfer.py
HIST_HELD_13_15 = {fold: {'s12': f'runs/LC4B_{fold}', 's13': f'runs/SIMSAT_{fold}', 's14': f'runs/SIMSAT2_{fold}',
                          's15': f'runs/SIMSAT2D4F_{fold}'} for fold in ('upperRe', 'AoA15', 'AoA20')}

# Recorded reference values (full release, recipe seed; from the historical report.json / job.json / resume_*.pt / ZIPs).
_SIM_ALL = {'sim_digest': '3fe81944c168c4a18b9f1c04d472f1af08fb865106048742592cf3616afb8ffb', 'sim_samples': 147707,
            'sim_unique_windows': 95139, 'sim_windows': 95139, 'sim_calls': 18464,
            'sim_first_sample_ids': [['17775_20.h5', 353], ['26700_5.h5', 681], ['20325_0.h5', 56], ['19050_0.h5', 492],
                                     ['26700_10.h5', 263], ['7575_15.h5', 909], ['17775_5.h5', 94], ['13950_0.h5', 239]]}
HISTORICAL_13_15 = {
    's13': {'streams': {
                'finetune.batch_digest': 'e41595029352b5541ea8c937d9ced0b9bbad7665ccd6a7a494330b0241898ef1',
                'finetune.augmentation_digest': 'e4b10e3875fdd5a19b4ebd7d97ae1fb3fcaba5f99ef8aff881db1e2d4bf1ccd1',
                'finetune.lattice_phase_counts': [37105, 36922, 36677, 37004], 'finetune.cadence_counts': [112658, 35050],
                'finetune.cadence_trailing_fallbacks': 1795, 'finetune.unique_sampled_windows': 65926,
                'finetune.source_samples': 147708,
                'checkpoint.9232.batch_digest': 'de316e3c34bd4315a9ea247e6f9c0fcc22858d513d7b22700adff0922bd0e531',
                'checkpoint.9232.augmentation_digest': 'f75249573a23d46be33ca4db864f5f0808a280a271538b9f0d90521c256f9ed9',
                **_SIM_ALL},
            'initial_states': {'point': '2e43a25f00eb174f08a5210167ba7ca2225659ef3ab22aba0eb85d31f02f4e63',
                               'heads': '9954a49cd7acf907a09a826b053f8617650e0645836f97060514cc6aca4b126e'},
            'teacher_point_hash': '2e43a25f00eb174f08a5210167ba7ca2225659ef3ab22aba0eb85d31f02f4e63',
            'deployed_states': {'model.pt': '43e5763fba2a757338e257a3569ca15d443d555b97006150f6e967a8f467517a',
                                'center_head.pt': '6d1d0edca8015a0846fb6428b439db40633748e4c6d88a999dc8387e3ecaf56e',
                                'radius_head.pt': '509b40a02e3eec0c90190b2ecaca3407b566615e3a545442ff37a78c0b81a0b2',
                                'hidden_head.pt': 'bed3f5b72bec7542350e137ebd067c4890271b1ee5a8746712d91525a4f0b4b4',
                                'observed_center_head.pt': '1fae8011c0cac5bcaa9152184110f20ef5b86b030936fce311ebc744aa8e1508'},
            'zip_sha256': 'bbbd6c1a852643721bd903cfccc1ba3bde241c4976fb6d83ffab83ebb0184c3f',
            'checkpoint_sha256': '5eca83e1432b3f35f3ec5b1cc00575fa6f3c66eb417f44f9de9a370db1936be9',
            'plan_sha256': '2f82860a99ddc141d2e26078f42cbce4da2e4933f98acf662e4fa4dc06bf94c7',
            'training_loop_seconds': 17205.2, 'peak_device_bytes': 4197913088, 'peak_host_bytes': 8620490752},
    's14': {'streams': {
                'finetune.batch_digest': '973e5c2649cae61ef2380ab1a3670f3c930011c9e523f5534780be140b7d6b3a',
                'finetune.augmentation_digest': 'f74e4005af12a27f579e9bebbbfbf617e4c4f498d4ec433a0778ed1b20a1d630',
                'finetune.lattice_phase_counts': [36523, 37110, 37056, 37019], 'finetune.cadence_counts': [112393, 35315],
                'finetune.cadence_trailing_fallbacks': 1790, 'finetune.unique_sampled_windows': 65926,
                'finetune.source_samples': 147708,
                'checkpoint.9232.batch_digest': '12d0f35a8615a4e462bc352540059fd3e768bb1df0f225465162afe43c2882be',
                'checkpoint.9232.augmentation_digest': '948c6a4e589ce5ad89623641cd6aa966dca270c8ab191318fd76d2e33ef220df',
                **_SIM_ALL},
            'initial_states': {'point': '43e5763fba2a757338e257a3569ca15d443d555b97006150f6e967a8f467517a',
                               'heads': '5c398dca934b2ab9c4ef1939fb03e5968c2662a1d92c1b277fbfa170c785c36b'},
            'teacher_point_hash': '43e5763fba2a757338e257a3569ca15d443d555b97006150f6e967a8f467517a',
            'deployed_states': {'model.pt': '2a3bac849ef279725176940a099a89ad8148b558d63302dd07e02a37aec2a99c',
                                'center_head.pt': '49017a6831a9d9fc422ca84f420c5c484db1c399ddd60b1a849af1b19f82e4f4',
                                'radius_head.pt': '33ae7e4db78087e3bd896ed14f8dbc97a368889e04acfd022795d8995559327a',
                                'hidden_head.pt': '7e2b82395683935940ec7207a8da5d54a4dcdc66a7d9bc207e674f3f5519c103',
                                'observed_center_head.pt': 'e456d2a037abffca59d592cdc78b076d6354606cb6ef80af0c53d8670816a738'},
            'zip_sha256': 'a0db2cc3f308a286317e22a7b646a8ea69676c53993f9218573e442dcc6fe146',
            'checkpoint_sha256': '26cdbb7067de0d03bd5acca3e048eb957e54ede52d3346d6b509f5f058278e0f',
            'plan_sha256': '73103544f8aede051bf7c421924eb6348f64802d619f3257b4db9fe31815813d',
            'training_loop_seconds': 18259.3, 'peak_device_bytes': 4197913088, 'peak_host_bytes': 8584482816},
    's15': {'streams': {
                'distillation.batch_digest': '22ef532b2813b88cb6ba1ca9545e66ddea0da5f854b4012da0db7fdb5aa56753',
                'distillation.augmentation_digest': '8209f59c2f89db47925dbb52bd0a43eb262afb688c56ec2f88332177fd10d841',
                'distillation.lattice_phase_counts': [18471, 18244, 18518, 18621], 'distillation.cadence_counts': [56279, 17575],
                'distillation.cadence_trailing_fallbacks': 886, 'distillation.unique_sampled_windows': 65926,
                'distillation.source_samples': 73854,
                'finetune.batch_digest': '3afa43a561189ad5107e611d4fcc0ed3814316964be0ceb84c85c539e923f3f8',
                'finetune.augmentation_digest': 'f9006c88a392067c5bffab2fcfdaf64e893e922127306122c3291d4d0827f243',
                'finetune.lattice_phase_counts': [36894, 36769, 37032, 37013], 'finetune.cadence_counts': [112511, 35197],
                'finetune.cadence_trailing_fallbacks': 1752, 'finetune.unique_sampled_windows': 65926,
                'finetune.source_samples': 147708,
                'checkpoint.9232.batch_digest': '22ef532b2813b88cb6ba1ca9545e66ddea0da5f854b4012da0db7fdb5aa56753',
                'checkpoint.18464.batch_digest': 'dd92f2d6b34dbc45170584a4e37dc5edf89d61fe95d716bd3380feb5332e1c80',
                'checkpoint.18464.augmentation_digest': '0d2698ff30526966ee7a22662a9e598502c460996210b877d3598d86f6bb1c69',
                'sim_digest': 'ceb054143fdd652efd69f351158ce2f00dbade5a39a908695fc9753b9c7d15b0', 'sim_samples': 147712,
                'sim_unique_windows': 91154, 'sim_windows': 95139, 'sim_calls': 18464,
                'sim_first_sample_ids': [['15225_15.h5', 76], ['24150_5.h5', 30], ['27975_0.h5', 301], ['22875_5.h5', 513],
                                         ['3750_20.h5', 353], ['8850_0.h5', 272], ['11400_5.h5', 7], ['3750_5.h5', 448]]},
            'initial_states': {'point': '0b2ef221ae2227f22e05a0020926a732d29cdf96738a969d157568439cb388a9',
                               'heads': '2b4b47f4187318324a8b4b27cf9ee61af29a726f90e9d9f8c92e7e4cb8d99974'},
            'teacher_point_hash': '2a3bac849ef279725176940a099a89ad8148b558d63302dd07e02a37aec2a99c',
            'deployed_states': {'model.pt': 'c5744fa0a01f9808619e290cf7eecff99e29f0c0f0837a36e8ccbbbb909b29a4',
                                'center_head.pt': '14910bd9c6fd4c9ca63528d7e2daa07a132136cfbb3a066222c29b2f9e3cb281',
                                'radius_head.pt': 'be8f6681578d8a66bb451aa70cd22384f7f1584e6aa8e6e641be7d7a5489520f',
                                'hidden_head.pt': '4cf65d1c08d2358591a1ce119eeb7a837dea93fe3fefec705cb95f5b0c2ac7a9',
                                'observed_center_head.pt': '3309dcb10bd8f77172ad12c80933da8cc64f7b48c5127bda7eb53a4cdcddc25b'},
            'zip_sha256': 'a4813dc0dc6cdd14d630fda7988044ce39f60990166cf230e03e2e534f2c3c0b',
            'checkpoint_sha256': 'ec0e4e0619d7aaddefa1d7fc051b169738a9e76d40918bf6d1f7ed3a33b3dc1f',
            'plan_sha256': '912cee5911135b2cd7d04593cda5c54d61c05a14866512f599681842d4191deb',
            'training_loop_seconds': 22851.7, 'peak_device_bytes': 2307968512, 'peak_host_bytes': 8533540864},
}

# ---------------------------------------------------------------- GPU estimates (one RTX A5500), from historical logs
# 'logged' = historical full81 wall (training_loop_seconds of report.json) with the A5500 SHARED: S13/S14 ran as 4
# concurrent jobs (full81 + 3 held surfaces), S15 next to held-surface jobs. 'unshared':
#   simsat (S13/S14): no uncontended SIMSAT-type log exists; modelled from uncontended logs of the same components:
#     LCSD3H alone 0.0851 s/step (10-block student fwd+bwd + teacher fwd), SIMSAT2D4_full81 alone (4-block student,
#     identical S15 compute) 0.068 s/step distillation / 0.1136 s/step fine-tune -> teacher fwd ~0.014 s,
#     10-block student fwd+bwd ~0.071 s -> SIMSAT step = 2 student fwd+bwd (real + simulation) + 2 teacher fwd
#     ~0.170 s/step (SIMA_full81 fastest logged segment 0.142 s/step has one teacher fwd less). Uncertainty ~+-25 %.
#   d4f (S15): measured on runs/SIMSAT2D4_full81 (same trainer family, same model sizes, ran alone): 3,155 s for
#     9,232 + 18,464 steps -> 0.1139 s/step on average.
GPU_ESTIMATES_13_15 = {
    'simsat': {'per_step': 0.170, 'overhead': 100, 'logged': {'s13': 17206, 's14': 18260},
               'note': 'modelled (no uncontended log); logged 17,206 s (S13) / 18,260 s (S14) as 4 concurrent jobs; device peak '
                       '3.91 GiB of the 4 GiB allocator cap set by train_simat.py (historical launches: PYTORCH_CUDA_ALLOC_CONF='
                       'expandable_segments:True, OOM retries when the GPU was shared); host peak 8.6 GB'},
    'd4f': {'per_step': 3155 / 27696, 'overhead': 0, 'logged': {'s15': 22852},
            'note': 'measured on SIMSAT2D4_full81 (identical compute, alone): 3,155 s; logged S15 22,852 s shared; device peak '
                    '2.3 GB; host peak 8.5 GB'},
    'package': {'note': 'CPU ~5-15 s per stage (two checkpoints, five torch.save payloads, deterministic ZIP)'},
    'final': {'note': 'CPU seconds (F5 + MA member transforms, deterministic ZIPs) + ~10 s CPU runtime smoke'},
}

# Free GPU memory required before a launch (MiB): the historical launch gates of these trainers (launch_simsat.sh /
# chain_simsat3_full81.sh 4800 for the 10-block student; chain_e6.sh 3500 for the 4-block student). The trainers cap their
# own allocator at 4 GiB (set_per_process_memory_fraction); the CUDA context comes on top.
GPU_FREE_MIB_REQUIRED = {'simsat': 4800, 'd4f': 3500}


def estimate_seconds(kind, steps, key=None):
    """[unshared estimate, historical logged wall scaled to `steps` (None if never run at full81)]."""
    e = GPU_ESTIMATES_13_15[kind]
    logged = e['logged'].get(key)
    full = 27696 if kind == 'd4f' else 18464
    return [round(e['per_step'] * steps + e['overhead']), None if logged is None else round(logged * steps / full)]


# ---------------------------------------------------------------- static helpers
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def script_closure_13_15(script):
    """Static first-party import closure of a stage 13-15 trainer, including its simulation module (the trainers put
    runs/SIMALL on sys.path at run time, train_simat.py:78) and the plan's reflection_source (loaded by path)."""
    return sorted(set(import_closure([script, SIM_OF[script]], [(ROOT / script).parent, ROOT / 'scripts', ROOT / 'runs/SIMALL']))
                  | {SD10_REFLECTION})


def verified_source(path, expected):
    data = (ROOT / path).read_bytes()
    if sha256_bytes(data) != expected:
        raise SystemExit(f'code pin mismatch: {path}')
    return data.decode()


def lifted_function(path, expected, name):
    """Compile ONE top-level function from a pinned source file (the module itself is not imported, so its unrelated
    imports never run). Used for depth_distillation_cycle_v2.depth_runtime."""
    tree = ast.parse(verified_source(path, expected))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise SystemExit(f'{path}: expected one function {name}')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f'{path}:{name}', 'exec'), namespace)
    return namespace[name]


def f5_constants():
    """make_fixed5.py:14-22,29: `old`, `new`, `tail` and the two batch-size replacement pairs, read from the pinned file."""
    tree = ast.parse(verified_source(F5_SOURCE, PACKAGING_PINS[F5_SOURCE]))
    consts = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    pairs = [ast.literal_eval(n.iter) for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
             and [getattr(e, 'id', None) for e in n.target.elts] == ['a_', 'b_']]
    assert len(pairs) == 1 and len(pairs[0]) == 2, 'make_fixed5.py replacement pairs not found'
    return {'old': consts['old'], 'new': consts['new'], 'tail': consts['tail'], 'pairs': [list(p) for p in pairs[0]],
            'async_install': '_install_async_transfer(_TransferNamespace(globals()))'}


def ma_constants(suffix='MA'):
    """make_malloc.py:18-41: BLOCK (+ ARENA inserted after the M_TOP_PAD line for suffix MA) and the anchor line."""
    tree = ast.parse(verified_source(MA_SOURCE, PACKAGING_PINS[MA_SOURCE]))
    consts = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    points = [ast.literal_eval(n.args[0]) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == 'replace' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'BLOCK']
    assert len(points) == 1, 'make_malloc.py ARENA insertion point not found'
    block = consts['BLOCK']
    if suffix == 'MA':
        assert block.count(points[0]) == 1
        block = block.replace(points[0], points[0] + consts['ARENA'])
    return {'block': block, 'anchor': consts['anchor'], 'suffix': suffix}
