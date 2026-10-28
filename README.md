# Distilled Neural Operators for Efficient Sim-to-Real Flow Forecasting with Prediction Intervals

This is the solution of team **syouya_tobita** to Track 1 (Sim2Real) of the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/).

From 20 measured frames of the velocity field around a NACA4418 airfoil (PIV, 32 × 64), the model forecasts the next 20 frames, together with a prediction interval for every value. It is trained on paired CFD simulations and real measurements.

| Development leaderboard (submission 948431) | Composite | RelL2 | TKE | MVPE | Time | SPS |
|---|---|---|---|---|---|---|
| score | **82.033** | 94.66 | 79.77 | 94.12 | 93.43 | 44.32 |

The final ranking is decided on the organizers' private test set.

## Method

- **Point forecaster** ([dno/model.py](dno/model.py)): a hybrid neural operator. Each block combines:
  - a depthwise 5×5 convolution for local structure;
  - a low-rank Fourier mix for global structure (an FNO layer through a 32-channel bottleneck);
  - a gated FFN.

  A two-scale U-Net context is added. The decoder predicts the last frame plus a trend plus a learned correction, twice: one prediction gives the *direction* of the fluctuations and the other their *energy*.
- **Prediction intervals** ([dno/intervals.py](dno/intervals.py)): four small heads that see only the current window. They read parameter-free diagnostics of the window and the forecast, and the forecaster's last hidden field. They were trained with a smooth surrogate of the competition's Safe Prediction Score.
- **Training** ([docs/pipeline.md](docs/pipeline.md)): 15 fixed stages, with no validation-based selection.
  - Simulation pre-training, then real-data training of the point model and the intervals.
  - Composite-score training with exact sub-sampling and reflection augmentation ([dno/losses.py](dno/losses.py), [dno/augment.py](dno/augment.py)).
  - Self-distillation cycles, then simulation replay with a reflection-averaged teacher.
  - Finally, distillation into a 4-block student (2.6 M parameters) that runs in about 3.6 ms per window on the evaluation server.
- **TKECAL** ([dno/tkecal.py](dno/tkecal.py)): at inference, the forecast fluctuations are rescaled per pixel toward the turbulent kinetic energy of the *observed* frames. This partly undoes the damping toward the conditional mean.

[docs/method.md](docs/method.md) gives every stage with its optimizer, schedule, seeds and the changes that mattered.

## Repository

| Path | What it is |
|---|---|
| [`dno/`](dno) | Readable reference implementation in plain PyTorch: model, intervals, TKECAL, inference, objectives, augmentation |
| [`tests/`](tests) | Proof that `dno/` computes exactly what the submission computes |
| [`docs/`](docs) | Method description and the 15-stage training pipeline |
| [`package/`](package) | The submitted training package, which reproduces submission 948431 bit for bit (see [package/README.md](package/README.md)) |

`dno/` is the code to read. `package/` is the code that was run: its structure serves exact reproduction of the submission, not readability.

## Usage

**Train** (about 6 hours on one GPU; writes `/path/to/output/runtime`):
```bash
cp /path/to/starting_kit/scoring.py package/vendor/starting_kit_v9/   # official scoring program, not redistributed
docker build -t dno-package package/
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/data:/data:ro -v /path/to/output:/output dno-package \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

**Predict** with the readable reference:
```python
import dno
predictor = dno.Predictor.from_runtime('/path/to/output/runtime', device='cuda')
out = predictor.predict(x)   # x: float32 (N, 20, 32, 64, 3) in physical units
out['prediction'], out['lower'], out['upper']
```
The competition runtime is also available: `package/solution.py:load_predictor`. It does the same computation, plus CUDA graphs and fused kernels for speed.

**Test** (Python 3.10, torch 2.2.2, pytest; CPU is enough):
```bash
python -m pytest tests -q
```

## Verification
- **Reference vs submitted code.** `dno/` was run against the runtime code of submission 948431, which ships in `package/` and is byte-checked by the tests. With random weights, the forecast, the hidden field and the intervals are bitwise equal on CPU.
- **Training objectives.** The metrics, the SPS surrogate, the composite loss, the reflection augmentation, the window sampling and the complete stage-15 objective were checked against the trainers' own code; all are bitwise equal.
- **Trained weights on 12 real windows (GPU).** The reference and the competition runtime differ by at most 7e-5, and the relative L2 between their forecasts is 1.9e-5. The competition runtime runs part of its FFN in half precision.
- **Retraining.** `package/` retrained from scratch inside its Docker image reproduced the weights and predictions of submission 948431 bit for bit on an RTX A5500.

## Citation

If you use this code, please cite the competition and the benchmark:
```bibtex
@inproceedings{wu2026neurips,
  title={Neur{IPS} 2026 Real{PDE} Competition: Scientific {ML} for Real-world Physical Systems},
  author={Tailin Wu and Wenhao Deng and Haodong Feng and Ruiqi Feng and Tianrun Gao and Peiyan Hu and Hongyuan Liu and Tengfei Xu and Xinyu Yang and Lei Bai and Zongyi Li and Johannes Brandstetter and Nils Thuerey and Rose Yu and Anima Anandkumar},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Competition Track},
  year={2026},
  url={https://openreview.net/forum?id=2FqmnAyc0G}
}
@inproceedings{hu2026realpdebench,
  title={RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data},
  author={Peiyan Hu and Haodong Feng and Hongyuan Liu and Tongtong Yan and Wenhao Deng and Tianrun Gao and Rong Zheng and Haoren Zheng and Chenglei Yu and Chuanrui Wang and Kaiwen Li and Zhi-Ming Ma and Dezhi Zhou and Xingcai Lu and Dixia Fan and Tailin Wu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=y3oHMcoItR}
}
```

## License
MIT (see [LICENSE](LICENSE)). The competition data and the official starting kit are not included and remain under their own terms.
