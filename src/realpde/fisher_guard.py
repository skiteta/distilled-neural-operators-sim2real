from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterSlice:
    name: str
    parameter: nn.Parameter
    offset: int
    length: int
    complex_value: bool


@dataclass(frozen=True)
class ParameterGroup:
    name: str
    slices: tuple[ParameterSlice, ...]
    dimension: int


def logical_layer(name: str) -> str:
    """Map HybridFlowOperator parameters to architecture-level layers."""
    if name in {"trend_scale", "interval_temperature"}:
        return "global_calibration"
    parts = name.split(".")
    if parts[0] == "blocks":
        return ".".join(parts[:2])
    if parts[0] == "multiscale":
        if len(parts) >= 2 and parts[1] in {
            "down1", "block1", "down2", "block2", "decode1", "decode0",
        }:
            return ".".join(parts[:2])
        return "multiscale.scale"
    if parts[0] in {"velocity_head", "width_head"} and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def parameter_groups(model: nn.Module) -> tuple[ParameterGroup, ...]:
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {}
    order: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = logical_layer(name)
        if group not in grouped:
            grouped[group] = []
            order.append(group)
        grouped[group].append((name, parameter))

    output: list[ParameterGroup] = []
    seen: set[int] = set()
    for group in order:
        offset = 0
        slices: list[ParameterSlice] = []
        for name, parameter in grouped[group]:
            identity = id(parameter)
            if identity in seen:
                raise RuntimeError(f"parameter assigned twice: {name}")
            seen.add(identity)
            length = int(parameter.numel()) * (2 if parameter.is_complex() else 1)
            slices.append(ParameterSlice(name, parameter, offset, length, parameter.is_complex()))
            offset += length
        output.append(ParameterGroup(group, tuple(slices), offset))
    expected = sum(int(parameter.numel()) * (2 if parameter.is_complex() else 1)
                   for parameter in model.parameters() if parameter.requires_grad)
    if sum(group.dimension for group in output) != expected:
        raise RuntimeError("parameter grouping is not exhaustive")
    return tuple(output)


def real_flatten(group: ParameterGroup, values: dict[str, torch.Tensor | None]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for item in group.slices:
        value = values.get(item.name)
        if value is None:
            value = torch.zeros_like(item.parameter)
        if item.complex_value:
            flat = value.reshape(-1)
            parts.extend((flat.real.float(), flat.imag.float()))
        else:
            parts.append(value.reshape(-1).float())
    return torch.cat(parts) if parts else torch.empty(0, device=next(iter(values.values())).device)


def assign_real_flat(group: ParameterGroup, vector: torch.Tensor) -> None:
    if int(vector.numel()) != group.dimension:
        raise ValueError(f"dimension mismatch for {group.name}")
    cursor = 0
    for item in group.slices:
        parameter = item.parameter
        count = int(parameter.numel())
        if item.complex_value:
            real = vector[cursor : cursor + count].reshape(parameter.shape)
            imag = vector[cursor + count : cursor + 2 * count].reshape(parameter.shape)
            value = torch.complex(real, imag).to(parameter.dtype)
            cursor += 2 * count
        else:
            value = vector[cursor : cursor + count].reshape(parameter.shape).to(parameter.dtype)
            cursor += count
        parameter.grad = value
    if cursor != group.dimension:
        raise RuntimeError(f"incomplete gradient assignment for {group.name}")


def add_real_direction(group: ParameterGroup, vector: torch.Tensor, scale: float) -> None:
    if int(vector.numel()) != group.dimension:
        raise ValueError(f"dimension mismatch for {group.name}")
    cursor = 0
    with torch.no_grad():
        for item in group.slices:
            parameter = item.parameter
            count = int(parameter.numel())
            if item.complex_value:
                real = vector[cursor : cursor + count].reshape(parameter.shape)
                imag = vector[cursor + count : cursor + 2 * count].reshape(parameter.shape)
                delta = torch.complex(real, imag).to(parameter.dtype)
                cursor += 2 * count
            else:
                delta = vector[cursor : cursor + count].reshape(parameter.shape).to(parameter.dtype)
                cursor += count
            parameter.add_(delta, alpha=float(scale))


def gradients_by_name(
    loss: torch.Tensor,
    model: nn.Module,
) -> dict[str, torch.Tensor | None]:
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
    return {name: value.detach() if value is not None else None
            for (name, _), value in zip(named, gradients)}


def collect_empirical_gradients(
    model: nn.Module,
    objective: nn.Module,
    dataset: Any,
    indices: Iterable[int],
    device: torch.device,
    *,
    log_every: int = 25,
) -> tuple[tuple[ParameterGroup, ...], dict[str, torch.Tensor], list[float]]:
    groups = parameter_groups(model)
    selected = [int(index) for index in indices]
    matrices = {
        group.name: torch.empty((len(selected), group.dimension), dtype=torch.float32)
        for group in groups
    }
    losses: list[float] = []
    model.eval().requires_grad_(True)
    for row, index in enumerate(selected):
        observed, target = dataset[index][:2]
        observed = observed.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        point, width = model(observed, return_interval=True)
        loss, _ = objective(point, target, width)
        values = gradients_by_name(loss, model)
        losses.append(float(loss.detach()))
        for group in groups:
            matrices[group.name][row].copy_(real_flatten(group, values).cpu())
        if row == 0 or (row + 1) % int(log_every) == 0 or row + 1 == len(selected):
            print({"stage": "fisher_gradient_collection", "sample": row + 1,
                   "samples": len(selected), "loss": losses[-1]}, flush=True)
    return groups, matrices, losses


def build_subspace(
    groups: tuple[ParameterGroup, ...],
    matrices: dict[str, torch.Tensor],
    device: torch.device,
    *,
    variance_threshold: float = 0.9,
    maximum_rank: int = 500,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    bases: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, Any] = {}
    for group in groups:
        matrix_cpu = matrices.pop(group.name)
        matrix = matrix_cpu.to(device)
        gram = matrix @ matrix.transpose(0, 1)
        gram = 0.5 * (gram + gram.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
        eigenvectors = eigenvectors.flip(1)
        total = eigenvalues.sum()
        tolerance = max(float(eigenvalues[0]) * 1e-7, 1e-20)
        positive = int((eigenvalues > tolerance).sum())
        if positive == 0 or float(total) <= 0.0:
            raise RuntimeError(f"degenerate Fisher group: {group.name}")
        # CUDA cumsum has no deterministic kernel in the deployed PyTorch.
        # Rank selection is tiny (N <= 500), so perform this scalar decision on CPU.
        cumulative = torch.cumsum(eigenvalues.detach().cpu(), dim=0) / float(total)
        rank = int(torch.searchsorted(
            cumulative, torch.tensor(float(variance_threshold))
        ).item()) + 1
        rank = min(rank, positive, int(maximum_rank))
        selected_values = eigenvalues[:rank]
        selected_vectors = eigenvectors[:, :rank]
        basis = matrix.transpose(0, 1) @ selected_vectors
        basis = basis / selected_values.sqrt().clamp_min(1e-20).unsqueeze(0)
        column_norms = torch.linalg.vector_norm(basis, dim=0).clamp_min(1e-20)
        basis = basis / column_norms.unsqueeze(0)
        gram_basis = basis.transpose(0, 1) @ basis
        identity = torch.eye(rank, device=device, dtype=gram_basis.dtype)
        orthogonality_error = float((gram_basis - identity).abs().max())
        if orthogonality_error > 5e-3:
            basis, _ = torch.linalg.qr(basis, mode="reduced")
            gram_basis = basis.transpose(0, 1) @ basis
            orthogonality_error = float((gram_basis - identity).abs().max())
        bases[group.name] = basis.detach().cpu()
        diagnostics[group.name] = {
            "dimension": group.dimension,
            "samples": int(matrix.shape[0]),
            "positive_rank": positive,
            "protected_rank": rank,
            "captured_fraction": float(cumulative[rank - 1]),
            "top_eigenvalue": float(eigenvalues[0]),
            "protected_last_eigenvalue": float(eigenvalues[rank - 1]),
            "trace": float(total),
            "orthogonality_max_abs": orthogonality_error,
        }
        print({"stage": "fisher_subspace", "group": group.name,
               **diagnostics[group.name]}, flush=True)
        del matrix_cpu, matrix, gram, eigenvalues, eigenvectors, basis
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return bases, diagnostics


def move_bases(bases: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: basis.to(device) for name, basis in bases.items()}


def project_values(
    groups: tuple[ParameterGroup, ...],
    values: dict[str, torch.Tensor | None],
    bases: dict[str, torch.Tensor],
    *,
    alpha: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    output: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for group in groups:
        original = real_flatten(group, values)
        basis = bases[group.name]
        coefficients = basis.transpose(0, 1) @ original
        removed = basis @ coefficients
        projected = original - float(alpha) * removed
        original_sq = float(torch.dot(original.double(), original.double()))
        removed_sq = float(torch.dot(removed.double(), removed.double()))
        projected_sq = float(torch.dot(projected.double(), projected.double()))
        residual_coefficients = basis.transpose(0, 1) @ projected
        diagnostics[group.name] = {
            "original_norm_sq": original_sq,
            "removed_norm_sq": removed_sq,
            "projected_norm_sq": projected_sq,
            "coefficient_norm_sq": float(torch.dot(coefficients.double(), coefficients.double())),
            "residual_coefficient_norm_sq": float(torch.dot(
                residual_coefficients.double(), residual_coefficients.double()
            )),
        }
        cursor = 0
        for item in group.slices:
            count = int(item.parameter.numel())
            if item.complex_value:
                real = projected[cursor : cursor + count].reshape(item.parameter.shape)
                imag = projected[cursor + count : cursor + 2 * count].reshape(item.parameter.shape)
                output[item.name] = torch.complex(real, imag).to(item.parameter.dtype)
                cursor += 2 * count
            else:
                output[item.name] = projected[cursor : cursor + count].reshape(
                    item.parameter.shape
                ).to(item.parameter.dtype)
                cursor += count
    return output, diagnostics


def project_model_gradients(
    model: nn.Module,
    groups: tuple[ParameterGroup, ...],
    bases: dict[str, torch.Tensor],
    *,
    alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    values = {name: parameter.grad for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    projected, diagnostics = project_values(groups, values, bases, alpha=alpha)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = projected[name]
    return diagnostics
