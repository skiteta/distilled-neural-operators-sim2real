"""Shared constants and small helpers for the FINAL_REBUILD stage 1-7 driver.

Stdlib only at import time (the driver process never imports torch; torch work runs in
CPU-only or GPU child processes).  Spec: runs/FINAL_REBUILD/spec_stages_1_to_7.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

DRIVER = Path(__file__).resolve().parent
ROOT = DRIVER.parents[2]                      # runs/FINAL_REBUILD/driver -> project root
DATA = ROOT / 'data' / 'extracted'
REBUILD_BASE = ROOT / 'runs' / 'FINAL_REBUILD'
RUNTIME_CODE_DIR = DRIVER / 'runtime_code'
EXCLUDED_REAL = '7575_0.h5'
RETAINED_REAL = '6300_0.h5'
CASE = re.compile(r'^(\d+)_(\d+)\.h5$')

# Spec section 1: environment for every stage (cwd=ROOT).  CUBLAS is mandatory
# because every stage enables torch.use_deterministic_algorithms(True).
ENV = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
       'OPENBLAS_NUM_THREADS': '4', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
       'CUDA_MODULE_LOADING': 'LAZY'}

# ---- code pins (asserted: code must equal the historical lineage code) --------------
C253_SCRIPT = 'scripts/run_c252_release_only_c180_scratch_recipe.py'
C253_CONFIG = 'configs/track1_final_retrain_c180.yaml'
T6 = 'runs/C758R_full81_exposure_matched/train.py'
T7 = 'runs/C787R_full81_point_preserved/train.py'
CODE_PINS = {
    C253_SCRIPT: 'e60d75477dc6c99c7c2f8a75df6c46f21b0a006b2e1289951629913291c73150',
    C253_CONFIG: '024ca548583921dff26a643227fd81c726c1b0d3f5228634e39109208d1c13b6',
    T6: '1f5de6c0f4f28243af568473a4153d009fe4f6ef8754439bc18e5c1588e17847',
    T7: '45d4959003cc854b2956c42fe1205e094633fea1ef832e1463a51ead43593bcf',
}
# Vendored runtime code (spec section 1 / 5a); copied with the *.py + LICENSE filter only.
RUNTIME_CODE = {
    'LICENSE': '0a6b51e48c63e6eb105b6aae1639bca755ee3d0b04cebab4b92d266527ebb22b',
    'factorized_model.py': '58be092335223326b2ecacd64d489a7fc87f3582e23131525233e09e838126fc',
    'realpde/__init__.py': '0d12c507525545b048e8961c722249f977dc9ee817a05162758c89f6bb52d2e9',
    'realpde/fast_interval.py': 'd45406ab13f1c512b5a63045ba2e6fcecef6842f6a054d980fea86acf1f859a2',
    'realpde/flow_diagnostics.py': '828cdf6b08f7bf6c0285885b7ddfa6148f3d8c66284ecbf8217f765b0b9f116d',
    'realpde/fused_bias_ffn_cuda.py': '7fd3043f888fece7bdaf44ff957fbf03b5401c180369829281ceba724ba0eaeb',
    'realpde/hidden_sps_interval.py': '5f2f20105837097373444cd1eee1132870e8da6bc2fb87c134c66d65e898b3b4',
    'realpde/model.py': '2e30152f3eff53b945044e35937ddee69605d5faa02155be5f1a6fefbe02ac6c',
    'realpde/sps_radius_capacity.py': 'ed9b8ffbf67e36b635b68278c6e39b108ca697e2d46ba5ed7ec213410184c729',
    'submission.py': 'bacbc5a73a0e058a880ac38934e8490da0e5cc80fbc818b19d62d23a8380a163',
}
HEAD_FILES = ('center_head.pt', 'radius_head.pt', 'hidden_head.pt', 'observed_center_head.pt')
COMPONENT_FILES = ('model.pt',) + HEAD_FILES
HEAD_PREFIXES = {'center_head.pt': 'center.', 'radius_head.pt': 'radius.',
                 'hidden_head.pt': 'hidden.', 'observed_center_head.pt': 'observed.'}
HEAD_KEYS = {'center': 'center_head.pt', 'radius': 'radius_head.pt',
             'hidden': 'hidden_head.pt', 'observed': 'observed_center_head.pt'}

# Static import closure of the training chain (recorded, not asserted; C675 pins src itself).
CLOSURE = [C253_SCRIPT, C253_CONFIG, T6, T7, 'vendor/starting_kit_v9/scoring.py'] + [
    f'src/realpde/{n}.py' for n in (
        'config', 'data', 'fast_interval', 'flow_diagnostics', 'losses', 'metrics', 'model', 'train',
        'identity_interval_hardening', 'fixed_sps_head_training', 'fixed_sps_capacity_training',
        'sps_radius_capacity', 'sps_interval_recipe', 'hidden_sps_interval', 'fixed_hidden_sps_training',
        'temporal_start_dataset', 'fixed_observed_center_training', 'c628_pipeline', 'c628_ram_data')]

# ---- frozen policy constants (spec S6/S7: C757R policy and C784R endpoint, inlined) -------
REFERENCE_WINDOWS = 42847      # C757R eligible windows
REFERENCE_STEPS = 6000
S6_SEED = 2026091727
C757_TRAINING = {
    'steps': None, 'schedule_steps': None, 'seed': S6_SEED, 'batch': 8, 'point_lr': 0.0001,
    'head_lr': 0.0032, 'EMA': None,
    'loss': 'RelL2+TKE+0.1MVPE+0.25early3-6softSPS+2.0energyNormalizedPhase+0.2late10',
    'tau': '.04→.004 over horizon12000', 'optimizer': 'AdamW decay1e-4, point/head norm clip1',
    'sampling': 'complete shuffled epochs, final partial batch retained', 'endpoint': 'EMA6000 only',
    'wave_weight': 2.0, 'late_weight': 0.2,
    'phase_formula': 'pc*(norm(tc,time+uv)/clamp(norm(pc,time+uv),1e-8)); supervised training loss only, no target at runtime.',
    'sps_weight': 6.0}
S7_ENDPOINT_FRACTION = (3, 4)  # C784R selected_endpoint='previous' -> 0.75 * schedule_steps

# ---- historical references (RECORDED ONLY, never asserted; full release only) -------------
HISTORICAL = {
    's1_point': '81ae463996da5f048e98d4137b0640b1c86afaeb05e998d9aa44d3a9cbabe179',
    's1_head': 'a852a1b4782162e803604f940217764c830cbb160ebe3296c0bf2f8e2e08fd83',
    's2_center': '05ea7caae8f15003d46fbc432ed68d15055ee40c50dea43c8a9f20d8b381cc54',
    's2_radius': 'f90819b39cd408b5f281e9fa48de00e9ad8397879a7904a7d23fdb2dcbc9ef25',
    's3_hidden': '69ace3be2e698842d602fe666cb8ddb9fdd976616e5a6baf8d91e4815d82662c',
    's4_observed': '7619581127fd00efec36df0d7b42501db4edfa925f088d6f807e46a2b6811caf',
    's5_point': '80ef50d5907128c031ea3374adef5ff6fda3fb4fe3e9f2aec5573d07cc279430',
    's5_windows': 65926,
    's6_ema_point': '32e98c8c7505c21a54292925904a8cc7b2c20c6061a1e258ab757a285e99ca0f',
    's6_ema_center': 'd5306db415f2407be6317b539e139049511fcb7b6c2f6a11f02aca1e95100b0c',
    's6_ema_radius': '759adb3c9b7566d020fda6931cee5d7474096a660a094f4be288dc70dbb1845e',
    's6_ema_hidden': '39091817c688b2a532f825aff0e77507a18ae22ce1f584e09807bcb84eefbab0',
    's6_ema_observed': '5114637918451a45599cdef9b8a561b361aca1af8998efdaa0ce7bcb758c2cea',
    's6_batch_digest': '2f3ec175abb22ffdfbdca2c435633fe39335899cf5040838505a17e1ed8fd16e',
    's6_steps': 9232,
    's7_center': 'f785456b3d24ecff141ad45d7779c0925ff28401f3f22aa6d57236e260a170f7',
    's7_radius': '734997b07152b4fd3deee84692aee5c44351592209f9aa10fcc02c79249b05df',
    's7_hidden': 'a0b6c05e3af1680c8c9102eccb2afebd57e2105538a449f91d6b2e23fe18c116',
    's7_observed': '992a1fe16a4c5aec186e8ea9a7074de02a6e8dd03ffd4dcdd7b76bac7df54da5',
    's7_steps': 13848,
}
# Historical artifacts used ONLY by --smoke as read-only stand-ins / parity references.
HIST_C628_PARENT = 'runs/C728R_full81_dense_submission/original_parent_runtime'
HIST_C728_DIR = 'runs/C728R_full81_dense_submission'
HIST_C758_TRAINING = 'runs/C758R_full81_exposure_matched/training_candidate'
HIST_C787_DIR = 'runs/C787R_full81_point_preserved'
HIST_C787H_ZIP = 'runs/C787R_full81_point_preserved/C787H_EMA13848_RESEARCH.zip'

# Spec section 10 planning estimates (one RTX A5500, unshared unless noted), seconds.
GPU_ESTIMATES = {
    's1_4_c675': {'seconds': [3300, 7200], 'note': 'C675 RAM path: sim+5 folds+head+joint 3.3k-7.2k s; S2 ~280 s, S3 ~380 s, S4 ~170 s; C675 upper_re split took 757+5494 s shared'},
    's6_c758h': {'seconds': [800, 3100], 'note': '9232 x 0.077 s/step + ~60 s; 3.1k s when the GPU was shared; needs >= 8 GiB device (set_per_process_memory_fraction)'},
    's7_c787h': {'seconds': [410, 410], 'note': '4616 x 0.077 s/step + load; packaging 2.5 s CPU'},
}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def rel(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def regime(name):
    match = CASE.fullmatch(name)
    if match is None:
        raise ValueError('invalid case name: ' + name)
    return tuple(map(int, match.groups()))


def release_names(domain):
    names = sorted(p.name for p in (DATA / domain).glob('*.h5'))
    return [n for n in names if n != EXCLUDED_REAL] if domain == 'train_real' else names


def sd_hash(state_dict) -> str:
    """identity_interval_hardening.state_hash over a saved state_dict mapping.

    Equal to state_hash(module) for a module whose state_dict was saved (verified on the
    historical S1-S7 files); lets records be computed without constructing modules.
    """
    import torch
    digest = hashlib.sha256()
    for name, value in state_dict.items():
        digest.update(name.encode('utf-8'))
        value = value.detach().cpu().contiguous()
        if value.is_complex():
            value = torch.view_as_real(value)
        data = memoryview(value.numpy()).cast('B')
        for start in range(0, len(data), 1 << 20):
            digest.update(data[start:start + (1 << 20)])
    return digest.hexdigest()


def compare(value, reference):
    """Recorded (never asserted) comparison with a historical reference."""
    return {'value': value, 'historical': reference, 'match': value == reference}


def cpu_env(base=None):
    env = dict(os.environ if base is None else base)
    env.update(ENV)
    env['CUDA_VISIBLE_DEVICES'] = ''
    return env
