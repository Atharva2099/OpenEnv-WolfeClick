from __future__ import annotations

from openenv.core.env_server import create_app

from env.models import WolfeClickAction, WolfeClickObservation
from env.server.environment import WolfeClickEnvironment

app = create_app(
    WolfeClickEnvironment,
    WolfeClickAction,
    WolfeClickObservation,
    env_name="openenv-wolfeclick",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

