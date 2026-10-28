from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .config import load_config
from .data import discover_cases, format_inventory
from .diffusion import (
    calibrate_latent_flow,
    evaluate_latent_flow,
    finalize_latent_flow_refit,
    refit_latent_flow_all_cases,
    refit_teacher_all_cases,
    train_latent_flow,
)
from .distill import (
    calibrate_distilled_uncertainty,
    evaluate_distilled_uncertainty,
    train_distilled_uncertainty,
)
from .metrics import error_to_score, score_batch
from .model import HybridFlowOperator, ModelConfig, count_parameters
from .package import package_submission
from .spectral_posterior import (
    calibrate_spectral_posterior,
    evaluate_spectral_posterior,
    train_spectral_interval,
    train_spectral_mode,
    train_spectral_posterior,
)
from .sps_ordinal import evaluate_sps_ordinal, train_sps_ordinal
from .temporal_transfer import evaluate_temporal_transfer, train_temporal_transfer
from .tke_corrector import evaluate_tke_map, train_tke_map
from .tke_diffusion import evaluate_tke_diffusion, train_tke_diffusion
from .train import (
    calibrate_checkpoint,
    choose_device,
    load_checkpoint,
    make_dataset,
    make_loader,
    save_checkpoint,
    train_stage,
    train_uncertainty,
    validate,
)


def doctor() -> None:
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, indent=2))


def inspect_data(root: Path) -> None:
    candidates = [root / "train_sim", root / "train_real", root / "numerical", root / "real"]
    found = False
    for candidate in candidates:
        cases = discover_cases(candidate)
        if cases:
            found = True
            print(f"\n[{candidate}] {len(cases)} cases")
            print(format_inventory(cases))
    if not found:
        cases = discover_cases(root)
        if not cases:
            raise FileNotFoundError(f"no HDF5 cases below {root}")
        print(format_inventory(cases))


def smoke(config: dict[str, Any]) -> Path:
    device = choose_device(config.get("device", "auto"))
    model_config = ModelConfig.from_dict(config.get("model", {}), config.get("data", {}))
    model = HybridFlowOperator(model_config).to(device)
    batch = torch.randn(
        1,
        model_config.in_steps,
        32,
        64,
        3,
        device=device,
    )
    target = torch.randn(
        1,
        model_config.out_steps,
        32,
        64,
        3,
        device=device,
    )
    prediction, half_width = model(batch, return_interval=True)
    loss = (prediction[..., :2] - target[..., :2]).square().mean() + half_width.mean()
    loss.backward()
    output = Path(config["submission"]["checkpoint"])
    save_checkpoint(
        output,
        model,
        config,
        stage="smoke",
        step=1,
        metrics={"loss": float(loss.detach()), "parameters": float(count_parameters(model))},
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "shape": list(prediction.shape),
                "parameters": count_parameters(model),
                "checkpoint": str(output),
            }
        )
    )
    return output


@torch.inference_mode()
def evaluate(config: dict[str, Any]) -> None:
    device = choose_device(config.get("device", "auto"))
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(config["train"]["real_checkpoint"])
    model = load_checkpoint(checkpoint, device).eval()
    dataset = make_dataset(config, "real", "validation")
    loader = make_loader(dataset, config, shuffle=False)
    metrics = validate(
        model,
        loader,
        device,
        max_batches=int(config["train"].get("val_batches", 64)),
    )
    mapped = {name: float(error_to_score(value)) for name, value in metrics.items()}
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    prediction, half_width = model(inputs, return_interval=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    public = score_batch(
        prediction,
        targets,
        elapsed_seconds=elapsed,
        lower=prediction - half_width,
        upper=prediction + half_width,
    )
    print(
        json.dumps(
            {
                "raw_errors": metrics,
                "mapped_scores": mapped,
                "batch_seconds": elapsed,
                "official_subscores": public.__dict__,
                "warning": (
                    "inferred_final_score matches public rows; organizer formula is unpublished"
                ),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="realpde")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")

    inspect_parser = subparsers.add_parser("inspect-data")
    inspect_parser.add_argument("--root", type=Path, default=Path("data/extracted"))

    for name in ("smoke", "evaluate", "calibrate", "package"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--stage", choices=("sim", "real"), required=True)
    spectral_train_parser = subparsers.add_parser("train-spectral-posterior")
    spectral_train_parser.add_argument("--config", required=True)
    spectral_train_parser.add_argument("--stage", choices=("sim", "real"), required=True)
    tke_train_parser = subparsers.add_parser("train-tke-map")
    tke_train_parser.add_argument("--config", required=True)
    tke_train_parser.add_argument("--stage", choices=("sim", "real"), required=True)
    transfer_train_parser = subparsers.add_parser("train-temporal-transfer")
    transfer_train_parser.add_argument("--config", required=True)
    transfer_train_parser.add_argument("--stage", choices=("sim", "real"), required=True)
    tke_diffusion_train_parser = subparsers.add_parser("train-tke-diffusion")
    tke_diffusion_train_parser.add_argument("--config", required=True)
    tke_diffusion_train_parser.add_argument("--stage", choices=("sim", "real"), required=True)
    uncertainty_parser = subparsers.add_parser("train-uncertainty")
    uncertainty_parser.add_argument("--config", required=True)
    for name in (
        "train-diffusion",
        "refit-diffusion-all",
        "finalize-diffusion-refit",
        "calibrate-diffusion",
        "evaluate-diffusion",
        "train-distilled",
        "calibrate-distilled",
        "evaluate-distilled",
        "calibrate-spectral-posterior",
        "evaluate-spectral-posterior",
        "train-spectral-interval",
        "train-spectral-mode",
        "evaluate-tke-map",
        "evaluate-temporal-transfer",
        "evaluate-tke-diffusion",
        "train-sps-ordinal",
        "evaluate-sps-ordinal",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "inspect-data":
        inspect_data(args.root)
    else:
        config = load_config(args.config)
        if args.command == "smoke":
            smoke(config)
        elif args.command == "train":
            train_stage(config, args.stage)
        elif args.command == "train-uncertainty":
            train_uncertainty(config)
        elif args.command == "train-spectral-posterior":
            train_spectral_posterior(config, args.stage)
        elif args.command == "train-diffusion":
            train_latent_flow(config)
        elif args.command == "refit-diffusion-all":
            refit_teacher_all_cases(config)
            refit_latent_flow_all_cases(config)
        elif args.command == "finalize-diffusion-refit":
            finalize_latent_flow_refit(config)
        elif args.command == "calibrate-diffusion":
            calibrate_latent_flow(config)
        elif args.command == "evaluate-diffusion":
            evaluate_latent_flow(config)
        elif args.command == "train-distilled":
            train_distilled_uncertainty(config)
        elif args.command == "calibrate-distilled":
            calibrate_distilled_uncertainty(config)
        elif args.command == "evaluate-distilled":
            evaluate_distilled_uncertainty(config)
        elif args.command == "calibrate-spectral-posterior":
            calibrate_spectral_posterior(config)
        elif args.command == "evaluate-spectral-posterior":
            evaluate_spectral_posterior(config)
        elif args.command == "train-spectral-interval":
            train_spectral_interval(config)
        elif args.command == "train-spectral-mode":
            train_spectral_mode(config)
        elif args.command == "train-tke-map":
            train_tke_map(config, args.stage)
        elif args.command == "evaluate-tke-map":
            evaluate_tke_map(config)
        elif args.command == "train-temporal-transfer":
            train_temporal_transfer(config, args.stage)
        elif args.command == "evaluate-temporal-transfer":
            evaluate_temporal_transfer(config)
        elif args.command == "train-tke-diffusion":
            train_tke_diffusion(config, args.stage)
        elif args.command == "evaluate-tke-diffusion":
            evaluate_tke_diffusion(config)
        elif args.command == "train-sps-ordinal":
            train_sps_ordinal(config)
        elif args.command == "evaluate-sps-ordinal":
            evaluate_sps_ordinal(config)
        elif args.command == "evaluate":
            evaluate(config)
        elif args.command == "calibrate":
            calibrate_checkpoint(config)
        elif args.command == "package":
            package_submission(config)


if __name__ == "__main__":
    main()
