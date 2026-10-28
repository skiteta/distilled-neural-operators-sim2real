"""Stage 5 (C728P): factorized conversion of the S1 point plus byte copies of the S2-S4 heads.

Lifted from runs/C728R_full81_dense_submission/prepare.py:16,21-41,55,57 (spec Stage 5).
Dropped: the READY-ZIP parent (:18-20), C727R/C700R/work-record reads and hooks
(:5-7,17,45-61) and the weight-byte pin at :43, which is replaced by a recorded
state-hash equality. Kept: the case-set data pin (:16), the *.py/LICENSE copy filter
(:22-24, so no weight file is copied), the runtime-model code pin (:25), the exact
missing-key set (:28) and the CPU bit-exact identity check (:32-37). CPU only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COMPONENT_FILES, DATA, EXCLUDED_REAL, HEAD_KEYS, RETAINED_REAL, ROOT,  # noqa: E402
                    RUNTIME_CODE, RUNTIME_CODE_DIR, read_json, rel, sha256, write_json)

SUPPORT_FRAMES = 40


def c728p_convert(parent, out5, train_cases, code_dir=RUNTIME_CODE_DIR, data_root=DATA):
    """parent: {'model.pt': S1 final_point, 'center_head.pt': S2 center_live, 'radius_head.pt':
    S2 capacity_live, 'hidden_head.pt': S3, 'observed_center_head.pt': S4} -> out5/common_runtime."""
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from realpde.data import FlowWindowDataset, discover_cases
    from realpde.hidden_sps_interval import load_hidden_residual
    from realpde.identity_interval_hardening import load_head, state_hash
    from realpde.model import HybridFlowOperator, ModelConfig
    from realpde.sps_radius_capacity import load_expanded_head

    parent = {name: Path(path).resolve(strict=True) for name, path in parent.items()}
    if set(parent) != set(COMPONENT_FILES):
        raise ValueError('parent must name exactly ' + ', '.join(COMPONENT_FILES))
    torch.set_num_threads(2)
    # prepare.py:16 data pin (release inventory), then restrict to this run's training cases.
    inventory = discover_cases(Path(data_root) / 'train_real', exclude_files=[EXCLUDED_REAL])
    names = [c.path.name for c in inventory]
    assert len(names) == 81 and RETAINED_REAL in names and EXCLUDED_REAL not in names
    train_cases = list(train_cases)
    cases = [c for c in inventory if c.path.name in set(train_cases)]
    assert [c.path.name for c in cases] == train_cases and len(set(train_cases)) == len(train_cases)
    full_release = train_cases == names

    out5 = Path(out5)
    out5.mkdir(parents=True, exist_ok=False)
    runtime = out5 / 'common_runtime'
    runtime.mkdir()
    code_dir = Path(code_dir)
    for source in sorted(code_dir.rglob('*')):                              # prepare.py:22-24
        if source.is_file() and (source.suffix == '.py' or source.name == 'LICENSE'):
            target = runtime / source.relative_to(code_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    copied = {str(p.relative_to(runtime)): sha256(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
    if copied != RUNTIME_CODE:                                              # vendored code pin
        raise RuntimeError('runtime code differs from the pinned vendored code')
    assert sha256(runtime / 'realpde/model.py') == sha256(ROOT / 'src/realpde/model.py')  # :25

    spec = importlib.util.spec_from_file_location('c728_initial_factorized', runtime / 'factorized_model.py')
    F = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = F
    spec.loader.exec_module(F)
    base_payload = torch.load(parent['model.pt'], map_location='cpu', weights_only=False)      # :27
    base = HybridFlowOperator(ModelConfig.from_dict(base_payload['model_config'])).eval()
    base.load_state_dict(base_payload['state_dict'])
    parent_point_state = state_hash(base)
    model = F.FactorizedFlowOperator(base.config).eval()
    res = model.load_state_dict(base_payload['state_dict'], strict=False)
    expected = {'amplitude_trend_scale', *('amplitude_head.' + k for k in model.amplitude_head.state_dict()),
                *('direction_residual_head.' + k for k in model.direction_residual_head.state_dict())}
    assert set(res.missing_keys) == expected and not res.unexpected_keys                     # :28
    model.amplitude_head.load_state_dict(model.velocity_head.state_dict())                    # :29
    model.direction_residual_head.load_state_dict(model.velocity_head.state_dict())
    with torch.no_grad():                                                                     # :30-31
        model.amplitude_trend_scale.copy_(model.trend_scale)
        last = [v for v in model.direction_residual_head.modules() if isinstance(v, torch.nn.Conv2d)][-1]
        last.weight.zero_()
        last.bias.zero_()
    identity = []
    with torch.inference_mode():                                                              # :33-37
        for c in [cases[0], cases[-1]]:
            ds = FlowWindowDataset([c], stride=20)
            for i in [0, len(ds) - 1]:
                x, _ = ds[i]
                a = base(x[None])
                b = model(x[None])
                assert torch.equal(a, b)
                identity.append({'case': c.path.name, 'start': i * 20, 'CPU_point_bitexact': True})

    loaders = {'center': lambda p: load_head(p, 'cpu'), 'radius': lambda p: load_expanded_head(p, 'cpu'),
               'hidden': lambda p: load_hidden_residual(p, 'cpu'), 'observed': lambda p: load_hidden_residual(p, 'cpu')}
    parent_components = {'model.pt': {'path': rel(parent['model.pt']), 'sha256': sha256(parent['model.pt']),
                                      'state_hash': parent_point_state}}
    for key, name in HEAD_KEYS.items():
        parent_components[name] = {'path': rel(parent[name]), 'sha256': sha256(parent[name]),
                                   'state_hash': state_hash(loaders[key](parent[name]))}
    base_payload['state_dict'] = model.state_dict()                                           # :38
    base_payload['C728'] = {'parent_components': parent_components, 'all81_fit': full_release,
                            'train_cases': len(cases), 'initial_identity_checks': identity}
    torch.save(base_payload, runtime / 'model.pt')
    initial = {'point': state_hash(model)}                                                    # :39
    chain_internal = {}
    for key, name in HEAD_KEYS.items():                                                       # :40-41
        shutil.copyfile(parent[name], runtime / name)
        initial[key] = state_hash(loaders[key](runtime / name))
        chain_internal[name + '_state_equals_parent'] = initial[key] == parent_components[name]['state_hash']
    # :43 replacement: recorded chain-internal equality instead of a historical byte pin.
    windows = sum(len(range(0, c.frames - SUPPORT_FRAMES + 1)) for c in cases)               # :55
    members = {str(f.relative_to(runtime)): sha256(f) for f in sorted(runtime.rglob('*')) if f.is_file()}
    write_json(out5 / 'initialization_report.json', {                                        # cf. :57
        'status': 'COMPLETE', 'parent_components': parent_components, 'CPU_identity_checks': identity,
        'initial_states': initial, 'runtime_members': members, 'train_cases': len(cases),
        'train_windows': windows, 'full_release': full_release, 'chain_internal_state_checks': chain_internal,
        'weight_pins': 'recorded only (spec section 0.1)'})
    write_json(out5 / 'dense_plan.json', {'train_cases': train_cases, 'eval_cases': [],       # cf. :56
                                          'sampling': {'stride': 1, 'support_frames': SUPPORT_FRAMES,
                                                       'windows': windows}})
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-json', type=Path, required=True, help='JSON {component file name: path}')
    parser.add_argument('--cases', type=Path, required=True, help='JSON list of training case names')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, default=RUNTIME_CODE_DIR)
    args = parser.parse_args(argv)
    parent = {k: (ROOT / v if not Path(v).is_absolute() else Path(v)) for k, v in read_json(args.parent_json).items()}
    runtime = c728p_convert(parent, args.out, read_json(args.cases), args.code_dir)
    report = read_json(args.out / 'initialization_report.json')
    print(json.dumps({'status': 'S5_COMPLETE', 'runtime': rel(runtime), 'train_cases': report['train_cases'],
                      'windows': report['train_windows'], 'point_state': report['initial_states']['point'][:16]}))


if __name__ == '__main__':
    main()
