from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import error_to_score, inferred_composite_score


def score_to_error(score: float) -> float:
    if not 0.0 < score <= 100.0:
        raise ValueError(f"score must be in (0, 100], got {score}")
    return 2.0 * (100.0 / score - 1.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_point_fingerprint(path: str | Path) -> str:
    """Hash only tensors that affect a packaged model's deterministic point output."""
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    model_type = checkpoint.get("model_type", "hybrid_flow_operator")
    state = checkpoint["state_dict"]
    if model_type == "distilled_uncertainty":
        prefix = "anchor."
    elif model_type == "phase_latent_flow":
        prefix = "teacher."
    elif (
        model_type == "masked_spectral_posterior"
        and float(checkpoint.get("model_config", {}).get("point_correction_scale", 1.0)) == 0.0
    ):
        prefix = "anchor."
    else:
        prefix = ""

    digest = hashlib.sha256()
    selected = 0
    for original_name in sorted(state):
        if prefix and not original_name.startswith(prefix):
            continue
        name = original_name[len(prefix) :] if prefix else original_name
        if name.startswith("width_head.") or name == "interval_temperature":
            continue
        tensor = state[original_name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        selected += 1
    if selected == 0:
        raise ValueError(f"no point tensors found in {path}")
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalScores:
    relative_l2: float
    tke: float
    mvpe: float
    time_score: float
    sps_score: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LocalScores:
        return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PublicScores:
    rel_l2_score: float
    tke_score: float
    mvpe_score: float
    time_score: float
    sps_score: float
    final_score: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PublicScores:
        return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})

    @classmethod
    def from_components(
        cls,
        rel_l2_score: float,
        tke_score: float,
        mvpe_score: float,
        time_score: float,
        sps_score: float,
    ) -> PublicScores:
        final = inferred_composite_score(rel_l2_score, tke_score, mvpe_score, time_score, sps_score)
        return cls(
            rel_l2_score=rel_l2_score,
            tke_score=tke_score,
            mvpe_score=mvpe_score,
            time_score=time_score,
            sps_score=sps_score,
            final_score=float(final),
        )


@dataclass(frozen=True)
class PublicAnchor:
    name: str
    artifact_sha256: str
    runtime_family: str
    local: LocalScores
    public: PublicScores
    point_fingerprint: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PublicAnchor:
        return cls(
            name=str(values["name"]),
            artifact_sha256=str(values["artifact_sha256"]),
            runtime_family=str(values["runtime_family"]),
            local=LocalScores.from_dict(values["local"]),
            public=PublicScores.from_dict(values["public"]),
            point_fingerprint=(
                str(values["point_fingerprint"])
                if values.get("point_fingerprint") is not None
                else None
            ),
        )


class PublicTransferModel:
    """Calibrate exact local v9 scores to observed Public distribution shift.

    Known artifacts are returned exactly. An unseen artifact receives a transfer
    estimate and a deliberately wide uncertainty band; hidden Public labels make
    exact equality for a never-submitted model unidentifiable.
    """

    def __init__(
        self,
        anchors: list[PublicAnchor],
        final_uncertainty: float = 1.5,
        local_protocol: dict[str, Any] | None = None,
    ) -> None:
        if len(anchors) < 2:
            raise ValueError("at least two Public anchors are required")
        self.anchors = anchors
        self.final_uncertainty = float(final_uncertainty)
        self.local_protocol = dict(local_protocol or {})
        self._by_hash = {anchor.artifact_sha256: anchor for anchor in anchors}
        self._by_point_fingerprint = {
            anchor.point_fingerprint: anchor
            for anchor in anchors
            if anchor.point_fingerprint is not None
        }
        self._error_ratios = {
            "relative_l2": statistics.median(
                score_to_error(anchor.public.rel_l2_score) / anchor.local.relative_l2
                for anchor in anchors
            ),
            "tke": statistics.median(
                score_to_error(anchor.public.tke_score) / anchor.local.tke for anchor in anchors
            ),
            "mvpe": statistics.median(
                score_to_error(anchor.public.mvpe_score) / anchor.local.mvpe for anchor in anchors
            ),
        }
        x = [anchor.local.sps_score for anchor in anchors]
        y = [anchor.public.sps_score for anchor in anchors]
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        denominator = sum((value - x_mean) ** 2 for value in x)
        if denominator <= 0.0:
            raise ValueError("Public anchors must have distinct local SPS values")
        self._sps_slope = (
            sum((local - x_mean) * (public - y_mean) for local, public in zip(x, y, strict=True))
            / denominator
        )
        self._sps_intercept = y_mean - self._sps_slope * x_mean

    @classmethod
    def load(cls, path: str | Path) -> PublicTransferModel:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [PublicAnchor.from_dict(item) for item in values["anchors"]],
            final_uncertainty=float(values.get("final_uncertainty", 1.5)),
            local_protocol=values.get("local_protocol"),
        )

    def is_known_artifact(self, artifact_sha256: str | None) -> bool:
        return artifact_sha256 in self._by_hash

    def require_protocol(
        self,
        *,
        split: str,
        sample_count: int,
        device: str,
        call_batch_size: int,
    ) -> None:
        actual = {
            "split": split,
            "sample_count": int(sample_count),
            "device": device,
            "call_batch_size": int(call_batch_size),
        }
        mismatches = {
            key: (self.local_protocol[key], actual[key])
            for key in actual
            if key in self.local_protocol and self.local_protocol[key] != actual[key]
        }
        if mismatches:
            name = self.local_protocol.get("name", "configured protocol")
            detail = ", ".join(
                f"{key}: expected {expected!r}, got {observed!r}"
                for key, (expected, observed) in mismatches.items()
            )
            raise ValueError(
                f"Public transfer requires local protocol {name!r}; {detail}. "
                "Local v9 scoring is valid, but it cannot be compared with these Public anchors."
            )

    def _time_estimate(self, family: str) -> float:
        matches = [
            anchor.public.time_score for anchor in self.anchors if anchor.runtime_family == family
        ]
        if not matches:
            matches = [anchor.public.time_score for anchor in self.anchors]
        return float(statistics.median(matches))

    def estimate(
        self,
        local: LocalScores,
        *,
        runtime_family: str,
        artifact_sha256: str | None = None,
        point_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        if artifact_sha256 in self._by_hash:
            anchor = self._by_hash[str(artifact_sha256)]
            return {
                "method": "exact_submitted_anchor",
                "anchor": anchor.name,
                "scores": anchor.public.__dict__,
                "final_low": anchor.public.final_score,
                "final_high": anchor.public.final_score,
            }

        point_anchor = self._by_point_fingerprint.get(point_fingerprint)
        if point_anchor is None:
            rel = float(error_to_score(local.relative_l2 * self._error_ratios["relative_l2"]))
            tke = float(error_to_score(local.tke * self._error_ratios["tke"]))
            mvpe = float(error_to_score(local.mvpe * self._error_ratios["mvpe"]))
        else:
            rel = point_anchor.public.rel_l2_score
            tke = point_anchor.public.tke_score
            mvpe = point_anchor.public.mvpe_score
        sps = max(0.0, min(100.0, self._sps_intercept + self._sps_slope * local.sps_score))
        estimated = PublicScores.from_components(
            rel,
            tke,
            mvpe,
            self._time_estimate(runtime_family),
            sps,
        )
        return {
            "method": (
                "two_anchor_public_transfer_with_exact_point"
                if point_anchor is not None
                else "two_anchor_public_transfer"
            ),
            "anchor": None,
            "point_anchor": point_anchor.name if point_anchor is not None else None,
            "scores": estimated.__dict__,
            "final_low": max(0.0, estimated.final_score - self.final_uncertainty),
            "final_high": min(100.0, estimated.final_score + self.final_uncertainty),
            "warning": (
                "The v9 scorer is exact, but an unseen Public target distribution is not. "
                "Use final_low, not the point estimate, for submission selection."
            ),
        }
