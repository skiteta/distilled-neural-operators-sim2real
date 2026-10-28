# Distilled Neural Operators for Efficient Sim-to-Real Flow Forecasting with Prediction Intervals

This is the solution of team **syouya_tobita** (Syouya Tobita) to Track 1 (Sim2Real) of the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/). The task is to forecast 20 future frames of real PIV velocity fields around a NACA4418 airfoil from 20 observed frames, together with prediction intervals.

This repository is the training package that the organizers retrain for the final evaluation, with internal work-log comments and local paths removed; the training code is unchanged. `train()` rebuilds our Codabench submission 948431 from scratch.

## Results
Development (public) leaderboard, submission 948431:

| Composite | RelL2 | TKE | MVPE | Time | SPS |
|---|---|---|---|---|---|
| **82.033** | 94.66 | 79.77 | 94.12 | 93.43 | 44.32 |

The final ranking is decided on the organizers' private test set.

## Method in brief
- **Backbone.** A hybrid spectral/convolutional neural operator. Each block combines a low-rank Fourier mix, a depthwise 5×5 convolution and a gated FFN, and a two-scale U-Net context is added. An amplitude/direction factorization of the predicted fluctuations follows.
- **Score-aware prediction intervals.** Four small heads predict the interval center shift and radius. They are trained with a smooth surrogate of the competition's Safe Prediction Score (SPS) and see only the current window.
- **Training lineage (15 fixed stages).**
  - S1–S4: simulation pre-training, real-data fine-tuning and interval-head training.
  - S6–S8: losses built on a differentiable surrogate of the competition composite score, with decimation-phase, temporal-cadence and reflection augmentation.
  - S9–S12: self-distillation cycles.
  - S13–S14: simulation replay with a reflection-averaged teacher.
  - S15: distillation into a 4-block student.
- **Turbulence-energy calibration (TKECAL).** At inference, the predicted fluctuations of each window are rescaled toward the turbulent kinetic energy of the observed input frames.
- **Speed.** The 4-block student runs as a replayed CUDA graph with a fixed batch of 5, with fused NVRTC kernels and pinned asynchronous transfers. It reaches a Time score of 93.4.

[METHOD.md](METHOD.md) gives every stage with its losses, optimizer, steps and seeds.

## How to run
**1. Data.** Download the competition training data ([Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)) into a directory that contains `train_real/` and `train_sim/`. `train_real/7575_0.h5` is excluded by the code, so it may be present or absent.

**2. Official scoring program.** The training pipeline pins (by sha256) the scoring program of the official Track 1 starting kit, which is not redistributed here. Copy `scoring.py` from the starting kit to `vendor/starting_kit_v9/scoring.py` (see [vendor/starting_kit_v9/README.md](vendor/starting_kit_v9/README.md)).

**3. Build and train** (linux/amd64, one CUDA GPU):
```bash
docker build -t realpde-t1 .
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/data:/data:ro -v /path/to/output:/output realpde-t1 \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```
Training takes about 6 hours on one RTX A5500. PyTorch allocates at most 3.9 GiB of GPU memory, and the host needs about 12 GB of RAM plus about 25 GB of disk.

**4. Predict:**
```python
import solution
predictor = solution.load_predictor('/path/to/output', 'cuda')
out = predictor.predict(x)  # x: float32 (N, 20, 32, 64, 3) in physical units
# out['prediction'], out['lower'], out['upper']: float32 (N, 20, 32, 64, 3)
```

## Reproducibility
- Every stage runs with deterministic algorithms and fixed seeds.
- On our RTX A5500, `train()` inside the Docker image reproduced the weights, the runtime code and the predictions of submission 948431 bit for bit.
- On other GPUs the floating-point results differ, so the result is a new sample of the same recipe.

## Repository layout
| Path | Contents |
|---|---|
| `solution.py` | `train(data_dir, output_dir, device)` and `load_predictor(output_dir, device)` |
| `runs/FINAL_REBUILD/driver/` | stage drivers (S1–S7, S8–S12, S13–S15 and packaging) |
| `runs/*/`, `scripts/`, `src/realpde/`, `configs/` | the stage trainers and the library they use |
| `tkecal_patch.py` | the TKECAL inference transform |
| `code_948431.json` | sha256 of the runtime code of submission 948431, checked by `train()` |
| `submission/c628_runtime/` | runtime code of the first stages |

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
