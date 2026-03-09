"""OpenEnv-WolfeClick Battle Replay Viewer — Gradio HF Space app."""
from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

BATTLE_LOGS_DIR = Path(__file__).parent / "battle_logs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_battle_log(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def list_battle_logs() -> list[str]:
    if not BATTLE_LOGS_DIR.exists():
        return []
    return sorted(p.name for p in BATTLE_LOGS_DIR.glob("*.json"))


def format_action_badge(action: dict, was_illegal: bool) -> str:
    kind = action.get("action", "?")
    choice = action.get("choice", "?")
    icon = "[ATK]" if kind == "move" else "[SW]"
    badge = f"`{icon} {kind}: {choice}`"
    if was_illegal:
        badge += "  **[!] illegal -- fallback used**"
    return badge


def format_valid_actions(actions: list[dict]) -> str:
    parts = []
    for a in actions:
        icon = "[ATK]" if a["action"] == "move" else "[SW]"
        parts.append(f"{icon} {a['choice']}")
    return " | ".join(parts)


def reward_color(r: float) -> str:
    if r > 0:
        return "green"
    elif r < 0:
        return "red"
    return "gray"


def build_turn_block(turn: dict) -> str:
    t = turn["turn"]
    action = turn["chosen_action"]
    reward = turn["reward"]
    cumulative = turn["cumulative_reward"]
    illegal = turn.get("action_was_illegal", False)
    state_md = turn["state_markdown"]
    valid = turn.get("valid_actions", [])

    rc = reward_color(reward)
    sign = "+" if reward >= 0 else ""

    lines = [
        f"### Turn {t}",
        "",
        f"**Action:** {format_action_badge(action, illegal)}",
        "",
        f"**Reward:** <span style='color:{rc};font-weight:bold'>{sign}{reward:.2f}</span>"
        f" &nbsp;|&nbsp; **Cumulative:** {cumulative:.2f}",
        "",
        "<details><summary><b>Battle State</b> (click to expand)</summary>",
        "",
        state_md,
        "",
        "</details>",
        "",
        f"<details><summary><b>Legal Actions</b></summary>\n\n{format_valid_actions(valid)}\n\n</details>",
        "",
        "---",
    ]
    return "\n".join(lines)


def build_summary(log: dict) -> str:
    outcome = log.get("outcome", "unknown")
    total_reward = log.get("total_reward", 0)
    total_turns = log.get("total_turns", 0)
    model = log.get("model", "Unknown")
    fmt = log.get("format", "unknown")

    icon = "WIN" if outcome == "won" else ("LOSS" if outcome == "lost" else "DRAW")
    rc = reward_color(total_reward)
    sign = "+" if total_reward >= 0 else ""

    return (
        f"## {icon} Battle Result: **{outcome.upper()}**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Model | `{model}` |\n"
        f"| Format | `{fmt}` |\n"
        f"| Turns | {total_turns} |\n"
        f"| Total Reward | <span style='color:{rc};font-weight:bold'>{sign}{total_reward:.2f}</span> |\n"
    )


# ---------------------------------------------------------------------------
# Replay generator
# ---------------------------------------------------------------------------

def replay_battle(log_name: str, speed: float):
    """Yield turn-by-turn markdown replay."""
    if not log_name:
        yield "Select a battle log to replay."
        return

    log = load_battle_log(BATTLE_LOGS_DIR / log_name)
    turns = log.get("turns", [])
    if not turns:
        yield "No turns recorded in this battle."
        return

    output_parts: list[str] = []
    for turn in turns:
        output_parts.append(build_turn_block(turn))
        yield "\n".join(output_parts)
        time.sleep(max(0.1, speed))

    output_parts.append(build_summary(log))
    yield "\n".join(output_parts)


# ---------------------------------------------------------------------------
# Tab content
# ---------------------------------------------------------------------------

ENV_DESIGN_MD = """
# Environment Design

## State Format

Each turn, the model receives a structured markdown state with three sections:

### Part A: Active Field
Current active Pokemon for both sides — HP, status, ability, item, stat modifiers, and estimated opponent speed range.

### Part B: Full Self Roster
All 6 Pokemon on the player's team with HP, status, item, and all known moves (with type abbreviation and base power).

### Part C: Opponent History
Every opponent Pokemon revealed so far — last known HP, status, revealed moves, items, and abilities. Updated incrementally each turn.

---

## Action Space

The model must output exactly one JSON action:

```json
{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}
```

Up to **4 moves** and **5 switches** are available per turn. The environment validates the action against legal options and applies a **-10.0 penalty** for illegal actions (hallucinated moves/Pokemon).

---

## Reward Components

| Component | Signal | Range |
|---|---|---|
| Damage dealt | +1.0 per 10% opponent HP reduced | Positive |
| Damage taken | -1.0 per 10% self HP lost | Negative |
| Knockouts | +3.0 per opponent faint, -3.0 per self faint | +/- |
| Healing | +1.0 per 10% healed (capped at 3.0/battle) | Positive |
| Setup | +0.5 per stat stage gained (capped at 2.0/mon) | Positive |
| Opponent setup | -0.5 per opponent stat stage gained | Negative |
| Type effectiveness | +0.5 super effective, -1.0 immune | +/- |
| Passive damage | Incremental reward for hazards/weather | Positive |
| Status inflicted | -0.5 to -1.0 for status ailments on self | Negative |
| Illegal action | -10.0 for hallucinated actions | Negative |
| Step penalty | -0.05 per turn (anti-stall) | Negative |

---

## Training Pipeline

```
Base Model (Qwen3-4B-Instruct)
        |
  [JSON Warm-up SFT]  <-- establish legal action baseline
        |
  [Rollout Collection]  <-- live Pokemon Showdown battles
        |
  [GRPO Training]  <-- optimize policy on real trajectories
        |
  LoRA Checkpoint  --> Hugging Face Hub
```

The training uses **GRPO (Group Relative Policy Optimization)** on real rollout data collected from live Pokemon Showdown battles. Each rollout records the state, chosen action, and shaped reward, which GRPO uses to improve the policy beyond simple format compliance.

---

## Architecture

```
Pokemon Showdown (Node.js, port 8000)
        |  WebSocket
PokeEnvClient (async background loop)
  |-- RLPlayer (queue-driven, receives model actions)
  |-- RandomPlayer (opponent)
        |
PokemonShowdownEnv (synchronous wrapper)
  |-- state_formatter  (markdown state for LLM)
  |-- action_space     (JSON validation + matching)
  |-- reward calculator (shaped multi-component reward)
        |
OpenEnv Server (FastAPI, standard reset/step API)
```
"""

GET_STARTED_MD = """
# Get Started

## Installation

```bash
git clone https://github.com/Atharva2099/OpenEnv-WolfeClick.git
cd OpenEnv-WolfeClick
pip install -e .
```

## Run a local battle (random actions)

```bash
python examples/run_single_episode.py
```

## Watch a trained model battle

```bash
python examples/watch_model_battle.py --revision grpo-qwen3-4b-run3
```

## Record a battle to JSON

```bash
python record_battle.py --revision grpo-qwen3-4b-run3 --output battle_logs/my_battle.json
```

## Training (Colab)

Open `trainer.ipynb` in Google Colab with a GPU runtime. The notebook handles:
1. Starting a local Pokemon Showdown server
2. Collecting rollout trajectories from live battles
3. Training a LoRA adapter with GRPO
4. Saving the checkpoint to Hugging Face Hub

---

## Links

| Resource | Link |
|---|---|
| Model Weights | [Atharva2099/openenv-smogon-rl](https://huggingface.co/Atharva2099/openenv-smogon-rl) |
| Training Notebook | `trainer.ipynb` |
| Watch Battle Notebook | `watch_battle.ipynb` |
| Benchmark Notebook | `benchmarks/benchmark.ipynb` |

## Trained Checkpoints

| Checkpoint | Description |
|---|---|
| `grpo-qwen3-4b-run1` | First GRPO training run |
| `grpo-qwen3-4b-run2` | Second run with tuned reward shaping |
| `grpo-qwen3-4b-run3` | Third run, best performing |

## OpenEnv Integration

This environment follows the [OpenEnv](https://github.com/openenv) standard. The server exposes `reset()` and `step()` endpoints:

```python
from env.client import WolfeClickEnv

env = WolfeClickEnv(server_url="http://localhost:8001")
obs = env.reset()
obs = env.step({"action_json": '{"action": "move", "choice": "flamethrower"}'})
```
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    available_logs = list_battle_logs()
    default_log = available_logs[0] if available_logs else None

    with gr.Blocks(
        title="OpenEnv-WolfeClick",
    ) as app:
        gr.HTML(
            "<h1 class='main-title'>OpenEnv-WolfeClick</h1>"
            "<p class='subtitle'>Train LLMs to play competitive Pokemon Showdown with GRPO</p>"
        )

        with gr.Tabs():
            # ---- Tab 1: Battle Replay ----
            with gr.Tab("Battle Replay", id="replay"):
                gr.Markdown(
                    "Watch a pre-recorded battle played by the GRPO-trained model. "
                    "Each turn shows the battle state the model received, the action it chose, "
                    "and the shaped reward signal."
                )
                with gr.Row():
                    log_selector = gr.Dropdown(
                        choices=available_logs,
                        value=default_log,
                        label="Battle Log",
                        scale=3,
                    )
                    speed_slider = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        value=0.8,
                        step=0.1,
                        label="Replay Speed (seconds per turn)",
                        scale=2,
                    )
                    play_btn = gr.Button("Play Replay", variant="primary", scale=1)

                replay_output = gr.Markdown(
                    value="Click **Play Replay** to start.",
                    label="Battle",
                )

                play_btn.click(
                    fn=replay_battle,
                    inputs=[log_selector, speed_slider],
                    outputs=replay_output,
                )

            # ---- Tab 2: Environment Design ----
            with gr.Tab("Environment Design", id="design"):
                gr.Markdown(ENV_DESIGN_MD)

            # ---- Tab 3: Get Started ----
            with gr.Tab("Get Started", id="start"):
                gr.Markdown(GET_STARTED_MD)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
