from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch

LIMIT = 256 * 1024 * 1024
JST = timezone(timedelta(hours=9), name="JST")
TIMESTAMP_PREFIX = re.compile(r"^\d{8}_\d{6}_JST__")
ARTIFACT_NAME_PATTERN = re.compile(
    r"^\d{8}_\d{6}_JST__(?:UNVERIFIED|READY|DO_NOT_SUBMIT|SUBMITTED|ARCHIVE|SMOKE|RESEARCH)__.+\.zip$"
)


def timestamped_submission_path(
    path: str | Path, *, now: datetime | None = None
) -> Path:
    """Return a sortable JST-prefixed ZIP path without double-prefixing it."""
    output = Path(path)
    if output.suffix.lower() != ".zip":
        raise ValueError(f"submission artifact must be a .zip file: {output}")
    if TIMESTAMP_PREFIX.match(output.name):
        if not ARTIFACT_NAME_PATTERN.fullmatch(output.name):
            raise ValueError(f"timestamped artifact lacks a valid status: {output}")
        return output
    moment = datetime.now(tz=JST) if now is None else now
    if moment.tzinfo is None:
        raise ValueError("artifact timestamp must be timezone-aware")
    stamp = moment.astimezone(JST).strftime("%Y%m%d_%H%M%S_JST")
    return output.with_name(f"{stamp}__UNVERIFIED__{output.name}")


def package_submission(config: dict[str, Any]) -> Path:
    root = Path(__file__).resolve().parents[2]
    submission_config = config["submission"]
    checkpoint_path = Path(submission_config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    output = timestamped_submission_path(submission_config["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite submission artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="realpde-submit-") as temporary:
        stage = Path(temporary)
        shutil.copy2(root / "submission" / "submission.py", stage / "submission.py")
        shutil.copy2(root / "src" / "realpde" / "model.py", stage / "model.py")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("model_type") in {
            "frequency_conditioned_tke_diffusion",
            "sps_ordinal_uncertainty",
        }:
            shutil.copytree(root / "src" / "realpde", stage / "realpde")
        compact = {
            "format_version": checkpoint.get("format_version", 1),
            "model_type": checkpoint.get("model_type", "hybrid_flow_operator"),
            "model_config": checkpoint["model_config"],
            "state_dict": checkpoint["state_dict"],
            "submission_batch_size": int(submission_config.get("batch_size", 4)),
        }
        torch.save(compact, stage / "model.pt")
        unpacked = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
        if unpacked >= LIMIT:
            raise RuntimeError(
                f"unpacked submission is {unpacked / 2**20:.1f} MiB; limit is 256 MiB"
            )

        smoke = (
            "import sys,numpy as np;sys.path.insert(0,sys.argv[1]);"
            "import submission;r=submission.predict(np.zeros((1,20,32,64,3),np.float32));"
            "assert set(r)=={'prediction','lower','upper'};"
            "assert all(x.shape==(1,20,32,64,3) for x in r.values())"
        )
        subprocess.run([sys.executable, "-c", smoke, str(stage)], check=True, timeout=300)

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path.suffix != ".pyc":
                    archive.write(path, arcname=path.relative_to(stage))
    print(f"created {output} ({output.stat().st_size / 2**20:.1f} MiB compressed)")
    return output
