from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from openenv.core.env_server import Action, Observation, State


@dataclass
class WolfeClickAction(Action):
    """Single step action for the environment.

    This wraps the constrained JSON interface already used by the local env:

        {"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}
    """

    action_json: str


@dataclass
class WolfeClickObservation(Observation):
    """Markdown battle state plus metadata."""

    state_markdown: str
    # Whether the episode is finished (battle over or truncated).
    done: bool = False
    # Shaped reward from the environment.
    reward: float = 0.0
    # Free-form metadata mirrored from the underlying env's info dict.
    metadata: Dict[str, Any] | None = None


class WolfeClickState(State):
    """Thin wrapper around the core State model."""

    # We rely on the base State fields (episode_id, step_count).
    # Any extra per-episode bookkeeping can live on the environment itself.
    pass

