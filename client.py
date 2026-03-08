from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from env.models import WolfeClickAction, WolfeClickObservation, WolfeClickState


class WolfeClickEnv(EnvClient[WolfeClickAction, WolfeClickObservation, WolfeClickState]):
    """HTTP/WebSocket client for the WolfeClick OpenEnv environment."""

    def _step_payload(self, action: WolfeClickAction) -> Dict[str, Any]:
        # Matches the action schema expected by WolfeClickEnvironment.step.
        return {"action_json": action.action_json}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[WolfeClickObservation]:
        obs_payload = payload.get("observation", {})
        obs = WolfeClickObservation(
            state_markdown=obs_payload.get("state_markdown", ""),
            done=bool(obs_payload.get("done", payload.get("done", False))),
            reward=float(payload.get("reward", 0.0)),
            metadata=obs_payload.get("metadata") or {},
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> WolfeClickState:
        # Base State already defines episode_id and step_count; we pass through the payload.
        return WolfeClickState(**payload)


__all__ = ["WolfeClickEnv"]

