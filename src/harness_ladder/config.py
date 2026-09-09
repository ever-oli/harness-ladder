"""Model and power-flag configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


DEFAULT_MODEL_ID = "openbmb/MiniCPM5-2B"
OPTIONAL_MODEL_IDS = ("Qwen/Qwen2.5-3B-Instruct",)

# Canonical power identifiers (P0 is the base V0 loop; P1 is few-shot; P2–P12 are stubs).
ALL_POWERS: FrozenSet[str] = frozenset(
    {
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
        "P11",
        "P12",
    }
)

POWER_ORDER: tuple[str, ...] = tuple(f"P{i}" for i in range(13))


@dataclass(frozen=True)
class ModelConfig:
    """Primary serving target is MiniCPM5-2B; Qwen is optional later."""

    model_id: str = DEFAULT_MODEL_ID
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 256
    seed: int = 0
    reasoning_token_budget: int = 32


@dataclass(frozen=True)
class PowerFlags:
    """Cumulative feature flags. Rung N enables powers P0..PN."""

    enabled: FrozenSet[str] = field(default_factory=lambda: frozenset({"P0"}))

    def __post_init__(self) -> None:
        unknown = self.enabled - ALL_POWERS
        if unknown:
            raise ValueError(f"Unknown power flags: {sorted(unknown)}")

    @classmethod
    def for_rung(cls, rung: int) -> "PowerFlags":
        if rung < 0 or rung > 12:
            raise ValueError(f"rung must be in 0..12, got {rung}")
        enabled = frozenset(POWER_ORDER[: rung + 1])
        return cls(enabled=enabled)

    def is_on(self, power: str) -> bool:
        return power in self.enabled

    def as_csv(self) -> str:
        return ",".join(p for p in POWER_ORDER if p in self.enabled)
