---
title: OpenEnv-WolfeClick Environment
emoji: 🎮
colorFrom: blue
colorTo: slate
sdk: docker
app_port: 8001
tags:
 - openenv
 - pokemon
 - rl
 - multi-agent
---

# OpenEnv-WolfeClick

OpenEnv-WolfeClick is an OpenEnv-compatible environment for training LLMs in competitive Pokemon Showdown battles.

The core idea is simple: rock-paper-scissors already shows that cyclic matchups create nontrivial reasoning. Competitive Pokemon scales that into a much richer world with hidden information, constrained legal actions, long-term resource tradeoffs, and an active opponent. This repo turns that setting into a trainable environment with a clean `reset()` / `step()` loop, an OpenEnv server wrapper, and a Colab GRPO training workflow.

## What is here

- `src/smogon_rl/`
  - core environment logic, state formatting, action validation, reward shaping, and the poke-env client
- `env/`
  - OpenEnv server package exposing the environment through `env.server.app:app`
- `trainer.ipynb`
  - Colab notebook for rollout collection and GRPO training
- `watch_battle.ipynb`
  - Colab notebook for running one live watched battle
- `benckmarks/benchmark.ipynb`
  - quick checkpoint-vs-checkpoint benchmark notebook
- `openenv.yaml`
  - OpenEnv entrypoint config
- `Dockerfile`
  - HF Spaces / Docker deployment path for the OpenEnv server

## Environment design

Each turn, the model receives a structured markdown state containing:

- active self Pokemon
- active opponent Pokemon
- HP, status, item, ability, and stat modifiers
- full self roster and currently known moves
- revealed opponent history
- the exact legal actions for the turn

The model must output exactly one JSON action:

```json
{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}
```

This keeps the interface concrete and legally grounded. The environment validates the action, executes it in a real Showdown battle, and returns the next state, reward, and episode metadata.

## Reward

Rewards are shaped, but still tied to battle progress. The signal includes:

- damage dealt and damage taken
- knockouts and faint penalties
- healing value
- setup value and opponent setup penalties
- passive damage and status effects
- illegal action penalties
- small anti-stall / truncation penalties

The goal is to create a denser learning signal without turning the task into a toy proxy objective.

## Training workflow

The training path is:

1. start local Pokemon Showdown in Colab
2. collect real rollout trajectories from live battles
3. store prompt, chosen action, and environment reward
4. train a LoRA adapter with GRPO on those real trajectories
5. benchmark checkpoints against each other on the same env budget

The repo includes both the training notebook and a smaller watch notebook for running one live battle with a chosen checkpoint.

## OpenEnv package

Yes, this repo includes a real OpenEnv environment package.

The deployable server lives at:

- `env/server/app.py`
- `env/server/environment.py`
- `env/models.py`

The OpenEnv config points to:

```yaml
app: env.server.app:app
```

That package wraps the local `PokemonShowdownEnv` and exposes it through the OpenEnv server interface.

## Local usage

Install the local package:

```bash
python3 -m pip install -e .
```

Run a simple local episode:

```bash
python3 examples/run_single_episode.py
```

Run one watched model battle:

```bash
python3 examples/watch_model_battle.py --revision grpo-qwen3-4b-run2
```

## Colab usage

- `trainer.ipynb`: collect rollouts and train with GRPO
- `watch_battle.ipynb`: start Showdown, load a checkpoint, and run one live battle
- `benckmarks/benchmark.ipynb`: compare checkpoints quickly

These notebooks assume a GPU runtime for model inference/training.

## Deployment

The repo includes the files needed for an OpenEnv-style deployment:

- `openenv.yaml`
- `Dockerfile`
- `env/` package

The Docker image starts:

- local Pokemon Showdown on port `8000`
- the OpenEnv FastAPI server on port `8001`

## Current artifacts

- HF model repo: `Atharva2099/openenv-smogon-rl`
- adapter revisions: `grpo-qwen3-4b-run1`, `grpo-qwen3-4b-run2`

## Status

This is a working end-to-end environment and training repo:

- live battle rollouts work
- GRPO training on real trajectories works
- checkpoint benchmarking works
- the OpenEnv server package exists in-repo

The next useful polish steps are HF Spaces deployment validation, README/blog cleanup, and a short write-up/demo around the environment design and results.
