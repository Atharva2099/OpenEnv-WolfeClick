# OpenEnv-WolfeClick

OpenEnv-WolfeClick is a reinforcement learning environment and training workflow for competitive Pokemon battles with large language models.

The project was built for the OpenEnv hackathon to answer a specific question: can an LLM learn to act in a partially observable, adversarial, long-horizon environment where legal actions are constrained, rewards are delayed, and the opponent is another agent?

This repo focuses on that environment and a minimal Colab training path.

## Why I Built This

Pokemon battles are a strong multi-agent training environment for LLMs because they require:

- hidden information and opponent modeling
- long-horizon planning over many turns
- legal action grounding under a constrained action space
- adapting to a changing world state after every action
- balancing local rewards against later consequences

I built this environment to make those properties trainable with a simple `reset()` / `step()` loop and a small JSON action interface.

## What is in this repo

- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl): environment, state formatting, action space, reward shaping, and client code
- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/trainer.ipynb`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/trainer.ipynb): main Colab notebook for warm-up SFT, rollout collection, and GRPO training
- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/examples`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/examples): small local examples
- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/pyproject.toml`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/pyproject.toml): package metadata

## Environment design

### State design

The state is not a raw simulator dump. It is a structured markdown representation designed to preserve strategic information while remaining readable to an LLM.

Each prompt includes:

- active self Pokemon
- active opponent Pokemon
- HP, status, ability, item, and current stat modifiers
- full self team roster with currently known moves
- opponent history and revealed information
- exact legal actions available this turn

This is implemented through the environment wrapper and state formatter:

- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/openenv_sync_env.py`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/openenv_sync_env.py)
- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/state_formatter.py`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/state_formatter.py)

My design goal was to expose enough information for strategic decisions without giving the model shortcuts that bypass the game structure.

### Action design

The action space is deliberately constrained.

The model must emit exactly one JSON object:

```json
{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}
```

At every step, legal actions are enumerated from the current battle state using:

- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/action_space.py`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/action_space.py)

This module does three important things:

- enumerates legal moves and switches for the turn
- builds the action instruction block shown to the model
- validates model outputs against the legal action set

This matters because I do not want the model to “sort of” describe an action. I want the environment to enforce a concrete legal interface.

### Reward design

The environment reward is shaped but still tied to battle outcomes.

Reward computation lives in:

- [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/reward.py`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/reward.py)

The reward includes:

- damage dealt to the opponent
- damage taken by the agent
- knockouts and faint penalties
- healing value
- setup value and opponent setup penalties
- passive damage value
- status penalties

The environment wrapper in [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/openenv_sync_env.py`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/src/smogon_rl/openenv_sync_env.py) adds practical rollout constraints:

- illegal action fallback handling
- illegal action penalties
- anti-stall living penalty
- battle length caps
- no-progress termination penalties

This separation is intentional:

- `reward.py` captures battle-quality shaping
- the env wrapper handles rollout hygiene and training throughput

## Training design

### 1. Warm-up SFT

The notebook begins with a supervised warm-up stage so the model learns to emit valid action JSON for the battle-state prompt format.

This does not claim strategic mastery. It only ensures the model is good enough to participate in the environment without collapsing into malformed outputs.

### 2. Real rollout collection

The policy is then run in real Pokemon Showdown battles. For each turn, the notebook stores:

- `prompt`
- `collected_action`
- `collected_reward`

This makes the rollout data usable for GRPO training while preserving the exact environment reward signal.

### 3. GRPO training

The GRPO reward used in the notebook is a wrapper around the stored rollout reward.

It is designed to preserve ranking pressure inside a completion group:

- malformed output is penalized strongly
- valid but different actions are penalized lightly
- the action matching the executed rollout action receives the collected environment reward plus a positive margin

That matters because raw rollout rewards alone do not always create a clean learning signal for group-relative optimization.

## How it works end to end

1. Start Pokemon Showdown locally in Colab.
2. Create the OpenEnv-style synchronous environment.
3. Format battle state into markdown.
4. Enumerate legal actions.
5. Generate one JSON action from the model.
6. Execute the action in the environment.
7. Receive next state, reward, done flag, and info.
8. Store rollout rows.
9. Train with GRPO on the collected rows.

## How to use

### Local package install

From the repo root:

```bash
python3 -m pip install -e .
```

### Colab training

Open [`/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/trainer.ipynb`](/Users/atharva/Desktop/Projects/OpenEnv-WolfeClick/trainer.ipynb) in Colab and run it top to bottom.

The notebook does the following:

1. clones or uses the repo
2. installs the training stack
3. loads the model and LoRA adapter
4. starts a local Pokemon Showdown server
5. runs JSON warm-up SFT
6. collects rollout data from real battles
7. trains with GRPO
8. optionally saves the adapter to Hugging Face Hub

### Requirements

- GPU runtime in Colab
- local Pokemon Showdown server started from the notebook
- Hugging Face token only if you want to push adapters

## Current status

This repo now has a working end-to-end path where:

- real battle rollouts are collected from the environment
- valid action JSON is produced reliably after warm-up
- GRPO can train on real rollout data in the non-quantized plain TRL path

This is the basis for my hackathon demo and benchmark runs.

## Submission notes

This repo is intended to be my clean hackathon submission repo.

Linked artifacts to add before submission:

- Hugging Face model repo
- Hugging Face Space using OpenEnv stable release `0.2.1`
- benchmark/results file
- 1-minute demo video
