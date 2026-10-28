"""RealPDE Track 1 (Sim2Real) training package: reproduces Codabench submission 948431 from scratch.

train(data_dir, output_dir, device)
    Runs the 15-stage training lineage with the stage drivers in runs/FINAL_REBUILD/driver:
    S1-S7 rebuild_1_7.py, S8-S12 rebuild_8_12.py, S13-S15 and packaging rebuild_13_15.py (S15 with the pre-C983 real
    loss, --loss-mode old). The final runtime is the packaged ZIP with the TKECAL inference transform applied to
    submission.py (tkecal_patch.py); its code is asserted to equal the code of submission 948431 byte for byte.
    The training data is first copied to <code root>/data/extracted (16 GB, removed at the end), the layout the
    lineage was trained with. Writes output_dir/runtime (everything inference needs) and output_dir/logs.
load_predictor(output_dir, device)
    Imports output_dir/runtime/submission.py (the Codabench API) and returns an object with predict(input_array):
    float32 (N, 20, 32, 64, 3) in physical units -> dict with prediction, lower and upper of the same shape.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = (  # (output directory under runs/FINAL_REBUILD, driver, extra arguments)
    ('R1_full', 'rebuild_1_7.py', []),
    ('R1_full_8_12', 'rebuild_8_12.py', ['--stage7', 'runs/FINAL_REBUILD/R1_full']),
    ('R1_full_13_15_E11R', 'rebuild_13_15.py', ['--stage12', 'runs/FINAL_REBUILD/R1_full_8_12', '--loss-mode', 'old']),
)
FINAL_ZIP = 'runs/FINAL_REBUILD/R1_full_13_15_E11R/final_simsat2d4fohf5ma/SIMSAT2D4FOHF5MA_FIXED5_REBUILT.zip'
RUNTIME_MODULES = ('realpde', 'factorized_model', 'async_transfer', 'fixed_transfer')
COPY_SKIP_PREFIXES = ('runs/FINAL_REBUILD/R1_', 'data/')
DOMAINS = ('train_real', 'train_sim')
STAGE_SPACE = 8 * 2**30  # stage outputs of the three drivers (about 3 GB) with margin


def _stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


class _Log:
    def __init__(self, path):
        self.file = open(path, 'a', buffering=1)

    def __call__(self, message):
        line = f'{_stamp()} {message}'
        print(line, flush=True)
        self.file.write(line + '\n')


def _data_files(data_dir):
    data = Path(data_dir).resolve()
    files = {}
    for domain in DOMAINS:
        if not (data / domain).is_dir():
            raise FileNotFoundError(f'{data / domain} is missing')
        files[domain] = sorted(p for p in (data / domain).iterdir() if p.is_file())
    return files


def _usable(root, need):
    """True if root/runs/FINAL_REBUILD is writable and its filesystem has `need` free bytes."""
    try:
        probe = root / 'runs/FINAL_REBUILD/.write_probe'
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text('ok')
        probe.unlink()
        return shutil.disk_usage(probe.parent).free >= need
    except OSError:
        return False


def _choose_root(output_dir, need, log):
    """The drivers write stage outputs under <root>/runs/FINAL_REBUILD and read <root>/data/extracted: use /workspace,
    or a copy of the code under output_dir/_workspace when /workspace is read-only or too small."""
    if _usable(ROOT, need):
        return ROOT
    root = Path(output_dir).resolve() / '_workspace'
    log(f'{ROOT} is not writable or has less than {need / 1e9:.0f} GB free; copying the code to {root}')
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT).as_posix()
        if (path.is_file() and not path.is_symlink() and not rel.startswith(COPY_SKIP_PREFIXES)
                and '__pycache__' not in rel):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if not _usable(root, need):
        raise OSError(f'neither {ROOT} nor {root} is writable with {need / 1e9:.0f} GB free')
    return root


def _copy_data(root, files, log):
    """Copy the release into <root>/data/extracted as regular files, as in the training environment of the lineage
    (the simulation-replay modules record data paths relative to the code root, so symlinks out of it cannot be used).
    7575_0.h5 is excluded by the code itself."""
    start, copied = time.time(), 0
    for domain, sources in files.items():
        target_dir = root / 'data/extracted' / domain
        if target_dir.is_symlink():
            target_dir.unlink()
        target_dir.mkdir(parents=True, exist_ok=True)
        names = {p.name for p in sources}
        stale = [p for p in target_dir.iterdir() if p.name not in names]
        if stale:
            raise FileExistsError(f'{target_dir} holds files that are not in the data directory: {stale[:5]}')
        for source in sources:
            target = target_dir / source.name
            if not (target.is_file() and target.stat().st_size == source.stat().st_size):
                shutil.copyfile(source, target)
                copied += source.stat().st_size
    log(f'training data copied under {root}/data/extracted: {sum(len(v) for v in files.values())} files, '
        f'{copied / 1e9:.1f} GB in {time.time() - start:.0f} s')


def _run_driver(root, name, script, extra, logs, log):
    out = root / 'runs/FINAL_REBUILD' / name
    command = [sys.executable, str(root / 'runs/FINAL_REBUILD/driver' / script), '--out', f'runs/FINAL_REBUILD/{name}', *extra]
    if out.exists():  # restart inside the same container: sealed stages are verified and reused
        command.append('--resume')
    log('RUN ' + ' '.join(command))
    start = time.time()
    with open(logs / f'{name}.console.log', 'a', buffering=1) as console:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            console.write(line)
            print(f'[{name}] {line}', end='', flush=True)
        code = process.wait()
    _save_driver_logs(out, logs / name)
    if code != 0:
        raise RuntimeError(f'{script} exited with {code}; see {logs}/{name}.console.log and {name}.driver.log')
    status = json.loads((out / 'rebuild_report.json').read_text()).get('status')
    if status != 'COMPLETE':
        raise RuntimeError(f'{script} finished with status {status}')
    log(f'OK {name} ({time.time() - start:.0f} s)')


def _save_driver_logs(out, target):
    """Keep the driver's text records (they live in the container otherwise): logs of every command, reports, plans."""
    if not out.is_dir():
        return
    for path in sorted(out.rglob('*')):
        rel = path.relative_to(out)
        if (path.is_file() and path.suffix in ('.log', '.json', '.jsonl', '.txt') and path.stat().st_size < 20_000_000
                and 'runtime' not in rel.parts and 'cpu_preflight' not in rel.parts):
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _install_runtime(root, runtime, log):
    sys.path.insert(0, str(ROOT))
    import tkecal_patch  # noqa: E402  (shipped next to this file)
    expected = json.loads((ROOT / 'code_948431.json').read_text())
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    with zipfile.ZipFile(root / FINAL_ZIP) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            if name == 'submission.py':
                data = tkecal_patch.apply(data.decode('utf-8')).encode('utf-8')
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for name, digest in expected.items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'runtime code {name} differs from the code of submission 948431')
    log(f'runtime written to {runtime} ({len(expected)} code files equal to submission 948431)')


def train(data_dir, output_dir, device):
    if not str(device).startswith('cuda'):
        raise ValueError(f'training needs a CUDA device, got {device!r}')
    output = Path(output_dir)
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = _Log(logs / 'train.log')
    start = time.time()
    log(f'START train(data_dir={data_dir}, output_dir={output_dir}, device={device}) python={sys.executable}')
    files = _data_files(data_dir)
    need = sum(p.stat().st_size for v in files.values() for p in v) + STAGE_SPACE
    root = _choose_root(output, need, log)
    _copy_data(root, files, log)
    for name, script, extra in STAGES:
        _run_driver(root, name, script, extra, logs, log)
    _install_runtime(root, output / 'runtime', log)
    shutil.rmtree(root / 'data/extracted', ignore_errors=True)  # the copy of the training data is no longer needed
    log(f'DONE training wall {time.time() - start:.0f} s')


class Predictor:
    def __init__(self, runtime, device):
        runtime = Path(runtime).resolve()
        if not (runtime / 'submission.py').is_file():
            raise FileNotFoundError(f'{runtime}/submission.py is missing; run train() first')
        # The runtime ships its own inference package `realpde` (not the training package in src/realpde).
        for module in list(sys.modules):
            if module.split('.')[0] in RUNTIME_MODULES:
                del sys.modules[module]
        if str(runtime) in sys.path:
            sys.path.remove(str(runtime))
        sys.path.insert(0, str(runtime))
        spec = importlib.util.spec_from_file_location('realpde_track1_submission', runtime / 'submission.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # loads the weights on CUDA when available (as on Codabench)
        self._module = module
        self.device = device

    def predict(self, input_array):
        return self._module.predict(input_array)


def load_predictor(output_dir, device):
    return Predictor(Path(output_dir) / 'runtime', device)
