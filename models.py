from __future__ import annotations

"""
Top-level models shim for the WolfeClick OpenEnv environment.

Exposes the same types as `env.models` so tools expecting a standard
OpenEnv layout (client.py, models.py, server/app.py) can import them.
"""

from env.models import WolfeClickAction, WolfeClickObservation, WolfeClickState

__all__ = ["WolfeClickAction", "WolfeClickObservation", "WolfeClickState"]

