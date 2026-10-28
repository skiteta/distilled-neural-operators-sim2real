from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import SIGMA_GLOBAL, safe_prediction_score
from .model import (
    HybridFlowOperator,
    MaskedSpectralPosteriorRefiner,
    OperatorBlock,
    SpectralPosteriorConfig,
)

# Keep this module importable in the official inference image.  The training
# stack eventually imports h5py, which is deliberately absent on Codabench.
# Training/evaluation helpers import that stack lazily in the functions below;
# the config and nn.Module definitions used by submission.py remain dependency
# free apart from torch.


@dataclass(frozen=True)
class SPSOrdinalConfig:
    """Conditional residual CDF optimized by the exact SPS Bayes action."""

    point: dict[str, Any]
    width: int = 64
    depth: int = 4
    spectral_rank: int = 16
    modes_y: int = 12
    modes_x: int = 24
    expansion: int = 2
    dropout: float = 0.02
    frequency_bins: int = 10
    threshold_count: int = 24
    threshold_minimum: float = 1e-3
    threshold_maximum: float = 3.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SPSOrdinalConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in dict(values).items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SPSOrdinalUncertaintyOperator(nn.Module):
    """Predict residual CDFs, then maximize the official expected SPS reward."""

    summary_channels = 26

    def __init__(self, config: SPSOrdinalConfig) -> None:
        super().__init__()
        self.config = config
        self.point = MaskedSpectralPosteriorRefiner(
            SpectralPosteriorConfig.from_dict(config.point)
        )
        self.point.requires_grad_(False)
        self.point.eval()
        self.out_steps = self.point.out_steps
        raw_phase_channels = 2 * self.point.in_steps * 2
        frequency_channels = 2 * config.frequency_bins * 2 * 2
        self.diagnostic_channels = (
            self.summary_channels + raw_phase_channels + frequency_channels
        )
        self.lift = nn.Conv2d(self.diagnostic_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        channels = self.out_steps * 2 * config.threshold_count
        self.cdf_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, channels, 1),
        )
        nn.init.zeros_(self.cdf_head[-1].weight)
        nn.init.zeros_(self.cdf_head[-1].bias)
        thresholds = torch.logspace(
            math.log10(config.threshold_minimum),
            math.log10(config.threshold_maximum),
            config.threshold_count,
        )
        self.register_buffer("thresholds", thresholds, persistent=True)

    def train(self, mode: bool = True) -> SPSOrdinalUncertaintyOperator:
        super().train(mode)
        self.point.eval()
        return self

    @staticmethod
    def _local_summaries(flow: torch.Tensor) -> torch.Tensor:
        temporal_mean = flow.mean(dim=1)
        fluctuation = flow - temporal_mean[:, None]
        variance = fluctuation.square().mean(dim=1)
        temporal_std = variance.clamp_min(1e-8).sqrt()
        local_tke = 0.5 * variance.sum(dim=-1, keepdim=True)
        recent = (flow[:, -1] - flow[:, -5]) / 4.0
        final_fluctuation = flow[:, -1] - temporal_mean
        return torch.cat(
            (temporal_mean, temporal_std, local_tke, recent, final_fluctuation),
            dim=-1,
        )

    def _frequency_features(self, flow: torch.Tensor) -> torch.Tensor:
        centered = flow - flow.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(centered.float(), dim=1, norm="ortho")
        bins = min(self.config.frequency_bins, spectrum.shape[1] - 1)
        selected = spectrum[:, 1 : bins + 1]
        if bins < self.config.frequency_bins:
            padding = selected.new_zeros(
                selected.shape[0],
                self.config.frequency_bins - bins,
                *selected.shape[2:],
            )
            selected = torch.cat((selected, padding), dim=1)
        packed = torch.view_as_real(selected)
        return packed.permute(0, 2, 3, 1, 4, 5).flatten(3)

    def _diagnostics(
        self, observed: torch.Tensor, point: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uv = observed[..., :2].float()
        point_uv = point[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_input = (uv - mean) / scale
        normalized_point = (point_uv - mean) / scale
        input_summary = self._local_summaries(normalized_input)
        point_mean = normalized_point.mean(dim=1)
        point_fluctuation = normalized_point - point_mean[:, None]
        point_variance = point_fluctuation.square().mean(dim=1)
        point_summary = torch.cat(
            (
                point_mean,
                point_variance.clamp_min(1e-8).sqrt(),
                0.5 * point_variance.sum(dim=-1, keepdim=True),
                normalized_point[:, -1] - normalized_input[:, -1],
            ),
            dim=-1,
        )
        batch, _, height, width, _ = observed.shape
        coordinates = HybridFlowOperator._coordinates(
            batch, height, width, observed.device, observed.dtype
        ).permute(0, 2, 3, 1)
        descriptors = torch.cat(
            (
                normalized_input[:, -1].mean(dim=(1, 2)),
                normalized_input[:, -1].std(dim=(1, 2)),
                normalized_point[:, -1].mean(dim=(1, 2)),
                normalized_point.std(dim=(1, 2, 3)),
            ),
            dim=-1,
        )
        descriptors = descriptors[:, None, None].expand(-1, height, width, -1)
        raw_phase = torch.cat((normalized_input, normalized_point), dim=1)
        raw_phase = raw_phase.permute(0, 2, 3, 1, 4).flatten(3)
        frequency = torch.cat(
            (
                self._frequency_features(normalized_input),
                self._frequency_features(normalized_point),
            ),
            dim=-1,
        )
        diagnostics = torch.cat(
            (
                input_summary,
                point_summary,
                coordinates,
                descriptors,
                raw_phase,
                frequency,
            ),
            dim=-1,
        )
        if diagnostics.shape[-1] != self.diagnostic_channels:
            raise RuntimeError(f"unexpected ordinal diagnostics: {diagnostics.shape}")
        return diagnostics.permute(0, 3, 1, 2).to(observed.dtype), scale

    def cdf_logits(
        self, observed: torch.Tensor, point: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, scale = self.features(observed, point)
        raw = self.cdf_head(hidden)
        logits = raw.reshape(
            observed.shape[0],
            self.out_steps,
            2,
            self.config.threshold_count,
            observed.shape[2],
            observed.shape[3],
        ).permute(0, 1, 4, 5, 2, 3)
        return logits.float(), scale

    def features(
        self, observed: torch.Tensor, point: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared complex-frequency features and the physical scale."""
        diagnostics, scale = self._diagnostics(observed, point)
        hidden = self.lift(diagnostics)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden), scale

    def decision_width(self, logits: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits).cummax(dim=-1).values
        physical = self.thresholds.view(1, 1, 1, 1, 1, -1) * scale[..., None]
        utility = probability * torch.exp(-2.0 * physical / SIGMA_GLOBAL)
        selected = utility.argmax(dim=-1, keepdim=True)
        return physical.expand_as(probability).gather(-1, selected)[..., 0]

    def forward(
        self, observed: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            point = self.point(observed)
        if not return_interval:
            return point
        logits, scale = self.cdf_logits(observed, point)
        width_uv = self.decision_width(logits, scale).to(point.dtype)
        width = torch.cat((width_uv, torch.zeros_like(width_uv[..., :1])), dim=-1)
        return point, width


def ordinal_loss(
    model: SPSOrdinalUncertaintyOperator,
    observed: torch.Tensor,
    target: torch.Tensor,
    values: dict[str, Any],
    *,
    point_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if point_override is None:
        with torch.no_grad():
            point = model.point(observed)
    else:
        point = point_override.detach()
    logits, scale = model.cdf_logits(observed, point)
    normalized_error = (point[..., :2] - target[..., :2]).abs() / scale
    cdf_target = normalized_error[..., None] <= model.thresholds
    mask = (target[..., :2] != 0.0)[..., None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits[mask], cdf_target.to(logits.dtype)[mask]
    )
    probability = torch.sigmoid(logits)
    monotonic = F.relu(probability[..., :-1] - probability[..., 1:])
    monotonic_mask = mask[..., :-1]
    monotonic_loss = monotonic[monotonic_mask].mean()

    physical = model.thresholds.view(1, 1, 1, 1, 1, -1) * scale[..., None]
    log_utility = torch.log(probability.clamp_min(1e-6)) - 2.0 * physical / SIGMA_GLOBAL
    temperature = float(values.get("decision_temperature", 0.20))
    policy = torch.softmax(log_utility / max(temperature, 1e-3), dim=-1)
    realized = cdf_target * torch.exp(-2.0 * physical / SIGMA_GLOBAL)
    decision_reward = (policy * realized).sum(dim=-1)
    measured = target[..., :2] != 0.0
    decision_loss = -decision_reward[measured].mean()
    loss = (
        float(values.get("cdf_weight", 1.0)) * bce
        + float(values.get("monotonic_weight", 0.10)) * monotonic_loss
        + float(values.get("decision_weight", 0.50)) * decision_loss
    )
    return loss, {
        "loss": float(loss.detach()),
        "cdf_bce": float(bce.detach()),
        "monotonic": float(monotonic_loss.detach()),
        "decision_reward": float((-decision_loss).detach()),
    }


def save_sps_ordinal_checkpoint(
    path: str | Path,
    model: SPSOrdinalUncertaintyOperator,
    config: dict[str, Any],
    *,
    step: int,
    metrics: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "sps_ordinal_uncertainty",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "stage": "sps_ordinal",
            "step": int(step),
            "metrics": metrics,
            "source_config": {
                key: value for key, value in config.items() if not key.startswith("_")
            },
        },
        output,
    )
    return output


def load_sps_ordinal_checkpoint(
    path: str | Path, device: torch.device
) -> SPSOrdinalUncertaintyOperator:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "sps_ordinal_uncertainty":
        raise ValueError(f"not an SPS ordinal checkpoint: {path}")
    model = SPSOrdinalUncertaintyOperator(
        SPSOrdinalConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def initialize_sps_ordinal(
    config: dict[str, Any], device: torch.device
) -> SPSOrdinalUncertaintyOperator:
    values = dict(config["sps_ordinal"])
    checkpoint = torch.load(values["point_checkpoint"], map_location=device)
    values["point"] = checkpoint["model_config"]
    model = SPSOrdinalUncertaintyOperator(SPSOrdinalConfig.from_dict(values)).to(device)
    model.point.load_state_dict(checkpoint["state_dict"], strict=True)
    model.point.requires_grad_(False)
    model.point.eval()
    return model


@torch.inference_mode()
def evaluate_sps_ordinal_model(
    model: SPSOrdinalUncertaintyOperator,
    dataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from .public_proxy import LocalScores, PublicTransferModel
    from .spectral_posterior import _score_candidate
    from .train import make_loader

    values = config["sps_ordinal"]
    loader = make_loader(
        dataset,
        config,
        shuffle=False,
        batch_size=int(values.get("validation_batch_size", 2)),
    )
    points: list[torch.Tensor] = []
    widths: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    regimes: list[tuple[float, float]] = []
    offset = 0
    model.eval()
    for observed, target in loader:
        count = observed.shape[0]
        observed = observed.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        point, width = model(observed, return_interval=True)
        points.append(point)
        widths.append(width)
        targets.append(target)
        regimes.extend(
            dataset.cases[dataset.windows[index][0]].regime
            for index in range(offset, offset + count)
        )
        offset += count
    point = torch.cat(points)
    width = torch.cat(widths)
    target = torch.cat(targets)
    runtime_score = float(values.get("runtime_score_assumption", 90.0))
    metrics = _score_candidate(
        point[..., :2], width[..., :2], target, runtime_score
    )
    case_sps: dict[str, float] = {}
    case_coverage: dict[str, float] = {}
    for regime in sorted(set(regimes)):
        indices = torch.tensor(
            [index for index, value in enumerate(regimes) if value == regime],
            device=device,
        )
        selected_point = point.index_select(0, indices)
        selected_width = width.index_select(0, indices)
        selected_target = target.index_select(0, indices)
        score, coverage = safe_prediction_score(
            selected_point,
            selected_target,
            lower=selected_point - selected_width,
            upper=selected_point + selected_width,
        )
        name = f"Re{regime[0]:g}_AoA{regime[1]:g}"
        case_sps[name] = float(score)
        case_coverage[name] = float(coverage)
    sps_values = list(case_sps.values())
    selection = (
        0.55 * metrics["sps_score"]
        + 0.30 * statistics.median(sps_values)
        + 0.15 * min(sps_values)
    )
    transfer = PublicTransferModel.load(
        values.get("public_transfer_config", "configs/public_transfer_v9.json")
    )
    public = transfer.estimate(
        LocalScores(
            relative_l2=metrics["relative_l2"],
            tke=metrics["tke"],
            mvpe=metrics["profile"],
            time_score=runtime_score,
            sps_score=metrics["sps_score"],
        ),
        runtime_family="one_pass",
    )
    return {
        **metrics,
        "case_sps": case_sps,
        "case_coverage": case_coverage,
        "case_median_sps": statistics.median(sps_values),
        "case_worst_sps": min(sps_values),
        "ordinal_selection": selection,
        "public_proxy_final": float(public["scores"]["final_score"]),
    }


def train_sps_ordinal(config: dict[str, Any]) -> Path:
    from .train import (
        _forever,
        choose_device,
        make_dataset,
        make_loader,
        seed_everything,
    )

    seed_everything(int(config.get("seed", 20260810)))
    device = choose_device(config.get("device", "auto"))
    values = config["sps_ordinal"]
    train_dataset = make_dataset(config, "real", "train")
    validation_dataset = make_dataset(config, "real", "validation")
    loader = make_loader(
        train_dataset,
        config,
        shuffle=True,
        batch_size=int(values.get("batch_size", 1)),
    )
    batches = _forever(loader)
    model = initialize_sps_ordinal(config, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    steps = int(values.get("steps", 2500))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(values.get("learning_rate", 2e-4)),
        weight_decay=float(values.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(values.get("minimum_learning_rate", 1e-6)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    )
    output = Path(values["checkpoint"])
    history = Path(values.get("history", "runs/track1_sps_ordinal/history.jsonl"))
    history.parent.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    log_every = int(values.get("log_every", 25))
    val_every = int(values.get("val_every", 250))
    throttle = max(0.0, float(values.get("throttle_seconds", 0.02)))
    model.train()
    for step in range(1, steps + 1):
        observed, target = next(batches)
        observed = observed.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss, components = ordinal_loss(model, observed, target, values)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, float(values.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if throttle:
            time.sleep(throttle)
        if step % log_every == 0 or step == 1:
            record = {
                "stage": "sps_ordinal",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **components,
            }
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % val_every == 0 or step == steps:
            metrics = evaluate_sps_ordinal_model(
                model, validation_dataset, config, device
            )
            print(
                json.dumps(
                    {"stage": "sps_ordinal", "step": step, "validation": metrics},
                    sort_keys=True,
                ),
                flush=True,
            )
            if float(metrics["ordinal_selection"]) > best:
                best = float(metrics["ordinal_selection"])
                save_sps_ordinal_checkpoint(
                    output, model, config, step=step, metrics=metrics
                )
            model.train()
    return output


@torch.inference_mode()
def evaluate_sps_ordinal(config: dict[str, Any]) -> dict[str, Any]:
    from .train import choose_device, make_dataset

    device = choose_device(config.get("device", "auto"))
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(config["sps_ordinal"]["checkpoint"])
    model = load_sps_ordinal_checkpoint(checkpoint, device).eval()
    dataset = make_dataset(config, "real", "validation")
    metrics = evaluate_sps_ordinal_model(model, dataset, config, device)
    result = {"checkpoint": str(checkpoint), "holdout": metrics}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result
