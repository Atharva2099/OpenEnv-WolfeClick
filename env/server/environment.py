from __future__ import annotations

import uuid
from typing import Any, Dict

from openenv.core.env_server import Environment

from env.models import WolfeClickAction, WolfeClickObservation, WolfeClickState
from smogon_rl.config import EnvConfig
from smogon_rl.openenv_sync_env import PokemonShowdownEnv


class WolfeClickEnvironment(Environment[WolfeClickAction, WolfeClickObservation, WolfeClickState]):
    """OpenEnv server wrapper around the local PokemonShowdownEnv."""

    def __init__(self) -> None:
        super().__init__()
        self._env = PokemonShowdownEnv(config=EnvConfig())
        # Underlying State tracks episode_id and step_count; we keep a separate battle counter.
        self._state = WolfeClickState(episode_id=str(uuid.uuid4()), step_count=0)
        self._battle_index: int = 0

    def reset(self, **kwargs: Any) -> WolfeClickObservation:
        """Start a new battle and return the initial observation."""
        self._battle_index += 1
        self._state = WolfeClickState(episode_id=str(uuid.uuid4()), step_count=0)
        state_str = self._env.reset()
        return WolfeClickObservation(
            state_markdown=state_str,
            done=False,
            reward=0.0,
            metadata={"battle_index": self._battle_index},
        )

    def step(self, action: WolfeClickAction, **kwargs: Any) -> WolfeClickObservation:
        """Apply one JSON action and return the next observation."""
        self._state.step_count += 1  # type: ignore[attr-defined]
        obs_str, reward, done, info = self._env.step(action.action_json)
        return WolfeClickObservation(
            state_markdown=obs_str,
            done=bool(done),
            reward=float(reward),
            metadata=info or {},
        )

    @property
    def state(self) -> WolfeClickState:
        return self._state

