"""CPU smoke of a packaged runtime directory (a rebuilt or historical deliverable, extracted): import its submission.py
exactly as the official loader would (runtime directory first on sys.path, its own realpde/ package, no project src/),
run predict() on released training windows (preflight_13_15.py runtime-input), and check the output contract.

Run it in a fresh interpreter with CUDA hidden (CUDA_VISIBLE_DEVICES=''): the runtime then takes its CPU paths
(fused kernels and transfer paths fall back; the F5/MA additions only replace the CUDA predict path and set glibc
malloc options). Output values on CPU are NOT the GPU values; the smoke checks importability, the depth-dependent
block-count assertion, shapes, dtype, finiteness, interval ordering, determinism of a repeated call and split invariance,
and records prediction hashes so that two runtimes with identical members can be compared.
Imports only the standard library, numpy and torch (never the driver modules or the project src/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='.npy [N,20,32,64,3] float32')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    runtime = args.runtime.resolve(strict=True)
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(runtime)] + [p for p in sys.path if p not in (here, '')]
    import numpy as np
    import torch
    assert not torch.cuda.is_available(), 'the runtime smoke must run with CUDA hidden'
    torch.set_num_threads(4)
    started = time.monotonic()
    import submission as S
    assert Path(S.__file__).resolve().parent == runtime, S.__file__
    import realpde
    assert Path(realpde.__file__).resolve().parent == runtime / 'realpde', realpde.__file__
    x = np.load(args.input)
    imported = time.monotonic() - started

    def digest(result):
        return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in sorted(result.items())}

    first = S.predict(x)
    second = S.predict(x)
    split = [S.predict(x[:1]), S.predict(x[1:])]
    joined = {k: np.concatenate([s[k] for s in split]) for k in first}
    checks = {
        'keys': sorted(first) == ['lower', 'prediction', 'upper'],
        'shapes': all(v.shape == x.shape for v in first.values()),
        'float32': all(v.dtype == np.float32 for v in first.values()),
        'finite': all(bool(np.isfinite(v).all()) for v in first.values()),
        'interval_ordered': bool(np.all(first['lower'] <= first['upper'])),
        'repeat_bit_identical': digest(first) == digest(second),
        'split_bit_identical': digest(first) == digest(joined),
    }
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'runtime': str(runtime), 'device': str(S._DEVICE),
              'point_depth': int(S._MODEL.config.depth), 'local_blocks': len(S._LOCAL_BLOCKS),
              'internal_batch_size': S._INTERNAL_BATCH_SIZE, 'transfer_group_size': S._TRANSFER_GROUP_SIZE,
              'fixed_transfer_installed': 'fixed_transfer' in sys.modules, 'input_shape': list(x.shape),
              'input_sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), 'checks': checks,
              'prediction_sha256': digest(first), 'import_seconds': round(imported, 2),
              'seconds': round(time.monotonic() - started, 2),
              'scope': 'CPU only: import + output contract; values are not the GPU values'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'point_depth', 'local_blocks', 'internal_batch_size', 'checks')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
