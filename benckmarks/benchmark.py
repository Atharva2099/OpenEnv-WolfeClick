from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from smogon_rl.action_space import (  # noqa: E402
    ActionOption,
    build_action_instructions,
    enumerate_actions,
    extract_action_json_from_text,
    parse_llm_action,
)
from smogon_rl.config import EnvConfig  # noqa: E402
from smogon_rl.openenv_sync_env import PokemonShowdownEnv  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = (
    "You are a competitive Pokemon battler. "
    "Analyse the battle state below and choose exactly one action. "
    "You MUST output ONLY a single JSON object — no explanation, no markdown fences, just the raw JSON.\n"
    '{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}'
)
MAX_NEW_TOKENS = 24
TEMPERATURE = 0.3
NUM_GENERATIONS = 4
BATTLES = 10
OUT_PATH = REPO_ROOT / "benckmarks" / "latest_results.md"
HF_TOKEN = os.environ.get("HF_TOKEN")


@dataclass
class CheckpointSpec:
    label: str
    mode: str  # json_sft | hub_adapter | local_adapter
    path_or_repo: Optional[str] = None
    revision: Optional[str] = None


CHECKPOINTS = [
    CheckpointSpec(
        label="Baseline (JSON SFT only)",
        mode="local_adapter",
        path_or_repo="./openenv_grpo_lora",
        revision=None,
    ),
    CheckpointSpec(
        label="Trained (GRPO)",
        mode="hub_adapter",
        path_or_repo="Atharva2099/openenv-smogon-rl",
        revision="grpo-qwen3-4b-run1",
    ),
]


@dataclass
class BenchmarkResult:
    label: str
    battles: int
    total_steps: int
    wins: int
    losses: int
    unknown: int
    format_hit_rate: float
    model_invalid: int
    env_illegal: int
    avg_reward_per_turn: float
    avg_battle_reward: float
    avg_battle_len: float
    wall_time_sec: float


def build_prompt_messages(state_str: str, valid_actions: list[ActionOption]) -> list[dict]:
    instructions = build_action_instructions(valid_actions)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{state_str}\n\n{instructions}"},
    ]


@torch.no_grad()
def generate_action_candidates(model, tokenizer, state_str: str, valid_actions: list[ActionOption]) -> list[str]:
    messages = build_prompt_messages(state_str, valid_actions)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=NUM_GENERATIONS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(out[input_len:], skip_special_tokens=True) for out in outputs]


def load_model(spec: CheckpointSpec):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    if spec.mode == "json_sft":
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    elif spec.mode == "hub_adapter":
        model = PeftModel.from_pretrained(
            model,
            spec.path_or_repo,
            revision=spec.revision,
            token=HF_TOKEN,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
    elif spec.mode == "local_adapter":
        model = PeftModel.from_pretrained(
            model,
            spec.path_or_repo,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
    else:
        raise ValueError(f"Unknown checkpoint mode: {spec.mode}")

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


def benchmark_checkpoint(spec: CheckpointSpec) -> BenchmarkResult:
    model, tokenizer = load_model(spec)
    env = PokemonShowdownEnv(
        config=EnvConfig(
            battle_format="gen4randombattle",
            verbose_logging=True,
            log_every_n_steps=10,
            poll_heartbeat_seconds=5.0,
        )
    )

    start = time.perf_counter()
    total_steps = 0
    total_reward = 0.0
    model_invalid = 0
    env_illegal = 0
    format_hits = 0
    battle_rewards: list[float] = []
    battle_lengths: list[int] = []
    wins = 0
    losses = 0
    unknown = 0

    for battle_idx in range(BATTLES):
        print(f"[{spec.label}] Battle {battle_idx + 1}/{BATTLES} started", flush=True)
        state_str = env.reset()
        done = False
        battle_reward = 0.0
        battle_len = 0

        while not done:
            battle = env._ensure_battle()
            valid_actions = enumerate_actions(battle)
            if not valid_actions:
                break

            candidates = generate_action_candidates(model, tokenizer, state_str, valid_actions)
            chosen_str = None
            for c in candidates:
                clean = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
                try:
                    parse_llm_action(clean, valid_actions)
                    chosen_str = clean
                    format_hits += 1
                    break
                except ValueError:
                    pass

                extracted = extract_action_json_from_text(c)
                if extracted is not None:
                    try:
                        parse_llm_action(extracted, valid_actions)
                        chosen_str = extracted
                        format_hits += 1
                        break
                    except ValueError:
                        pass

            if chosen_str is None:
                model_invalid += 1
                fb = valid_actions[0]
                chosen_str = json.dumps({"action": fb.action_type, "choice": fb.choice})

            state_str, reward, done, info = env.step(chosen_str)
            total_steps += 1
            battle_len += 1
            total_reward += float(reward)
            battle_reward += float(reward)
            if info.get("action_illegal", False):
                env_illegal += 1

        finished_battle = env._ensure_battle()
        won = getattr(finished_battle, "won", None)
        lost = getattr(finished_battle, "lost", None)
        if won is True:
            wins += 1
        elif lost is True:
            losses += 1
        else:
            unknown += 1

        battle_rewards.append(battle_reward)
        battle_lengths.append(battle_len)
        print(
            f"[{spec.label}] Battle {battle_idx + 1}/{BATTLES} finished | "
            f"steps={battle_len} reward={battle_reward:.3f}",
            flush=True,
        )

    wall = time.perf_counter() - start
    return BenchmarkResult(
        label=spec.label,
        battles=BATTLES,
        total_steps=total_steps,
        wins=wins,
        losses=losses,
        unknown=unknown,
        format_hit_rate=100.0 * format_hits / max(1, total_steps),
        model_invalid=model_invalid,
        env_illegal=env_illegal,
        avg_reward_per_turn=total_reward / max(1, total_steps),
        avg_battle_reward=statistics.mean(battle_rewards) if battle_rewards else 0.0,
        avg_battle_len=statistics.mean(battle_lengths) if battle_lengths else 0.0,
        wall_time_sec=wall,
    )


def to_markdown(results: list[BenchmarkResult]) -> str:
    lines = [
        "# Latest Benchmark Results",
        "",
        "| Checkpoint | Battles | Wins | Losses | Unknown | Format Hit Rate | Model Invalid | Env Illegal | Avg Reward / Turn | Avg Battle Reward | Avg Battle Length | Wall Time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.battles} | {r.wins} | {r.losses} | {r.unknown} | "
            f"{r.format_hit_rate:.1f}% | {r.model_invalid} | {r.env_illegal} | "
            f"{r.avg_reward_per_turn:.3f} | {r.avg_battle_reward:.3f} | {r.avg_battle_len:.1f} | {r.wall_time_sec:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = [benchmark_checkpoint(spec) for spec in CHECKPOINTS]
    markdown = to_markdown(results)
    print("\n" + markdown)
    OUT_PATH.write_text(markdown)
    print(f"Wrote benchmark results to {OUT_PATH}")


if __name__ == "__main__":
    main()
