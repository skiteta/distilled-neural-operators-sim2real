# Submission package (exact reproduction)

This directory is the training package of Codabench submission 948431, as retrained by the organizers. Two things were removed: internal work-log comments and local paths. The training code is unchanged.

`solution.py` implements the competition interface:
- `train(data_dir, output_dir, device)` runs the 15 stages described in [../docs/pipeline.md](../docs/pipeline.md) and writes `output_dir/runtime` and `output_dir/logs`.
- `load_predictor(output_dir, device).predict(x)` takes `x` as float32 `(N, 20, 32, 64, 3)` in physical units. It returns `prediction`, `lower` and `upper` of the same shape.

## Run

1. Put the official starting-kit `scoring.py` at `vendor/starting_kit_v9/scoring.py`; see [vendor/starting_kit_v9/README.md](vendor/starting_kit_v9/README.md).
2. Build the image and train (linux/amd64, one CUDA GPU, CUDA 12.1 wheels):

```bash
docker build -t dno-package package/
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/data:/data:ro -v /path/to/output:/output dno-package \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

`/path/to/data` holds `train_real/` and `train_sim/` from the competition release.

What `train()` uses:
- **Data:** it copies the data under `data/extracted` inside the container (16 GB) and deletes the copy at the end. The stage trainers record data paths relative to this root.
- **Disk:** stage intermediates take about 3 GB.
- **Run time and memory:** about 6 hours on one RTX A5500. PyTorch allocates at most 3.9 GiB of GPU memory, and peak host memory is 12 GB.

## Layout

| Path | Contents |
|---|---|
| `solution.py` | the competition interface |
| `tkecal_patch.py` | the TKECAL inference transform, applied to the packaged runtime |
| `code_948431.json` | sha256 of the runtime code of submission 948431; `train()` checks the result against it |
| `runs/FINAL_REBUILD/driver/` | the stage drivers: plans, code pins, stream replay, GPU launches, packaging |
| `runs/*/` | the stage trainers, unchanged from the runs that produced the submission (names are internal run names) |
| `src/realpde/`, `scripts/`, `configs/` | the training library (pinned as a whole; it also contains unused research modules) |
| `submission/c628_runtime/` | runtime code of the first stages |
| `Dockerfile` | python:3.10.16-slim-bookworm, torch 2.2.2+cu121 and pinned dependencies |

For a readable version of the model, the intervals, the objectives and the augmentation, see `../dno/`. `../tests/` shows that it computes the same thing as this package.
