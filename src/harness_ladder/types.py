"""Shared types for messages, trajectories, and task results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None


@dataclass
class Trajectory:
    """Full interaction for one task."""

    task_id: str
    messages: list[Message] = field(default_factory=list)
    final_answer: str = ""
    tokens_used: int = 0
    wall_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    success: bool
    expected: str
    actual: str
    trajectory: Trajectory
    notes: str = ""
