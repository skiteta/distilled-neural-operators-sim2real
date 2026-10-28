from __future__ import annotations

import torch

from .query_adaptive_temporal import QueryAdaptiveTemporalConfig, QueryAdaptiveTemporalHybrid


class QueryAdaptiveTrustRegionHybrid(QueryAdaptiveTemporalHybrid):
    """C396 model with a fixed scalar trust-region gain on adaptive updates."""

    def __init__(self, config: QueryAdaptiveTemporalConfig, gain: float) -> None:
        super().__init__(config)
        self.gain = float(gain)

    def _local_update(self, local, projection, size):  # type: ignore[no-untyped-def]
        return self.gain * super()._local_update(local, projection, size)

    def _film_parameters(self, global_context, index):  # type: ignore[no-untyped-def]
        gamma, beta = super()._film_parameters(global_context, index)
        return self.gain * gamma, self.gain * beta

