from __future__ import annotations

import math

import torch
from torch import nn

from .latent_state_refiner import ConvGRUCell, LatentStateRefiner, LatentStateRefinerConfig


class InverseLatentStateRefiner(LatentStateRefiner):
    """Latent forecaster with a training-only future teacher and inverse cycle."""

    def __init__(self, config: LatentStateRefinerConfig | None = None) -> None:
        super().__init__(config)
        self.inverse_transition = ConvGRUCell(3, self.config.latent_width)

    def _time_features(
        self,
        index: int,
        batch: int,
        height: int,
        width: int,
        source: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        time = (index + 1.0) / self.config.out_steps
        if reverse:
            time = -time
        phase = 2.0 * math.pi * time
        values = source.new_tensor((time, math.sin(phase), math.cos(phase)))
        return values[None, :, None, None].expand(batch, -1, height, width)

    def _encoded_sequence(
        self,
        uv: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        previous_frame: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        normalized = (uv.float() - mean) / std
        if previous_frame is None:
            previous = torch.cat((normalized[:, :1], normalized[:, :-1]), dim=1)
        else:
            previous_normalized = (previous_frame.float() - mean) / std
            previous = torch.cat((previous_normalized, normalized[:, :-1]), dim=1)
        temporal_delta = normalized - previous
        batch, steps, height, width, _ = normalized.shape
        coordinates = self._coordinates(batch, height, width, normalized)
        encoded = []
        for index in range(steps):
            frame = torch.cat(
                (
                    normalized[:, index].permute(0, 3, 1, 2),
                    temporal_delta[:, index].permute(0, 3, 1, 2),
                    coordinates,
                ),
                dim=1,
            )
            encoded.append(self.encoder(frame))
        return encoded

    def training_latent_paths(
        self, observed: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return student, privileged-future teacher, and reverse-cycle states."""
        observed_uv = observed[..., :2].float()
        target_uv = target[..., :2].float()
        mean = observed_uv.mean(dim=(1, 2, 3), keepdim=True)
        std = observed_uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        observed_encoded = self._encoded_sequence(observed_uv, mean, std)
        state = torch.zeros_like(observed_encoded[0])
        for frame in observed_encoded:
            state = self.assimilator(frame, state)
        past_state = state

        batch, _, latent_height, latent_width = state.shape
        predicted = state
        predicted_states = []
        for index in range(self.config.out_steps):
            features = self._time_features(
                index, batch, latent_height, latent_width, state
            )
            predicted = self.transition(features, predicted)
            predicted_states.append(predicted)

        target_encoded = self._encoded_sequence(
            target_uv, mean, std, previous_frame=observed_uv[:, -1:]
        )
        teacher = past_state.detach()
        teacher_states = []
        for frame in target_encoded:
            teacher = self.assimilator(frame, teacher)
            teacher_states.append(teacher)

        recovered = predicted_states[-1]
        recovered_states = []
        for reverse_index in range(self.config.out_steps):
            forward_index = self.config.out_steps - 1 - reverse_index
            features = self._time_features(
                forward_index,
                batch,
                latent_height,
                latent_width,
                state,
                reverse=True,
            )
            recovered = self.inverse_transition(features, recovered)
            recovered_states.append(recovered)

        return {
            "past_state": past_state,
            "predicted_states": torch.stack(predicted_states, dim=1),
            "teacher_states": torch.stack(teacher_states, dim=1),
            "recovered_states": torch.stack(recovered_states, dim=1),
        }

    def normalized_states(self, states: torch.Tensor) -> torch.Tensor:
        shape = states.shape
        flattened = states.reshape(-1, *shape[-3:])
        normalized = self.state_norm(flattened)
        return normalized.reshape(shape)

