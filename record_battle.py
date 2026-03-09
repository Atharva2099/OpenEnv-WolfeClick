"""Record a battle played by a trained LLM and save it as a JSON log.

Usage:
    python record_battle.py --revision grpo-qwen3-4b-run3 --output battle_logs/example_battle.json

Requires:
    - A local Pokemon Showdown server on port 8000 (or --showdown-dir set)
    - GPU recommended for model inference
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from smogon_rl.action_space import (
    build_action_instructions,
    enumerate_actions,
    extract_action_json_from_text,
    parse_llm_action,
)
from smogon_rl.config import EnvConfig
from smogon_rl.openenv_sync_env import PokemonShowdownEnv

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = (
    "You are a competitive Pokemon battler. "
    "Analyse the battle state below and choose exactly one action. "
    "You MUST output ONLY a single JSON object — no explanation, no markdown fences, just the raw JSON.\n"
    '{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a battle to JSON.")
    parser.add_argument("--repo-id", default="Atharva2099/openenv-smogon-rl")
    parser.add_argument("--revision", default="grpo-qwen3-4b-run3")
    parser.add_argument("--battle-format", default="gen4randombattle")
    parser.add_argument("--showdown-dir", default=os.environ.get("SHOWDOWN_DIR", ""))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--output",
        default="battle_logs/example_battle.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def _server_up(host: str = "127.0.0.1", port: int = 8000) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def ensure_showdown_server(showdown_dir: str) -> subprocess.Popen | None:
    if _server_up():
        return None
    if not showdown_dir:
        raise RuntimeError(
            "Pokemon Showdown is not running on port 8000 and --showdown-dir was not provided."
        )
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=showdown_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for _ in range(20):
        if _server_up():
            return proc
        time.sleep(0.5)
    raise RuntimeError("Pokemon Showdown did not start on port 8000.")


def build_prompt_messages(state_str: str, valid_actions) -> list[dict]:
    instructions = build_action_instructions(valid_actions)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{state_str}\n\n{instructions}"},
    ]


@torch.no_grad()
def generate_action_candidates(model, tokenizer, state_str, valid_actions, args):
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
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_generations,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(out[input_len:], skip_special_tokens=True) for out in outputs]


def load_model(repo_id: str, revision: str):
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=token, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model, repo_id, revision=revision, token=token,
        is_trainable=False, autocast_adapter_dtype=False,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


def choose_action(model, tokenizer, state_str, valid_actions, args):
    candidates = generate_action_candidates(model, tokenizer, state_str, valid_actions, args)
    for candidate in candidates:
        clean = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL).strip()
        try:
            parse_llm_action(clean, valid_actions)
            return clean, False
        except ValueError:
            pass
        extracted = extract_action_json_from_text(candidate)
        if extracted is not None:
            try:
                parse_llm_action(extracted, valid_actions)
                return extracted, False
            except ValueError:
                pass
    fallback = valid_actions[0]
    return json.dumps({"action": fallback.action_type, "choice": fallback.choice}), True


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    showdown_proc = ensure_showdown_server(args.showdown_dir)
    try:
        print("Loading model...")
        model, tokenizer = load_model(args.repo_id, args.revision)

        env = PokemonShowdownEnv(
            config=EnvConfig(
                battle_format=args.battle_format,
                max_steps_per_battle=args.max_steps,
                verbose_logging=True,
                log_every_n_steps=1,
                poll_heartbeat_seconds=5.0,
            )
        )

        print("Starting battle...")
        state_str = env.reset()
        battle = env._ensure_battle()

        turns = []
        done = False
        total_reward = 0.0
        step_idx = 0

        while not done:
            step_idx += 1
            battle = env._ensure_battle()
            valid_actions = enumerate_actions(battle)
            if not valid_actions:
                print("No valid actions available; ending battle.")
                break

            action_json_str, was_illegal = choose_action(
                model, tokenizer, state_str, valid_actions, args
            )
            valid_actions_json = [
                {"action": a.action_type, "choice": a.choice} for a in valid_actions
            ]

            prev_state = state_str
            state_str, reward, done, info = env.step(action_json_str)
            total_reward += float(reward)

            try:
                chosen_action = json.loads(action_json_str)
            except json.JSONDecodeError:
                chosen_action = {"raw": action_json_str}

            turn_record = {
                "turn": step_idx,
                "state_markdown": prev_state,
                "valid_actions": valid_actions_json,
                "chosen_action": chosen_action,
                "reward": round(float(reward), 4),
                "cumulative_reward": round(total_reward, 4),
                "action_was_illegal": was_illegal,
            }
            turns.append(turn_record)
            print(
                f"step={step_idx} action={action_json_str} "
                f"reward={reward:.3f} total={total_reward:.3f} done={done}"
            )

        final_battle = env._ensure_battle()
        won = getattr(final_battle, "won", None)
        lost = getattr(final_battle, "lost", None)
        outcome = "won" if won else ("lost" if lost else "unknown")

        battle_log = {
            "model": f"{MODEL_NAME} + LoRA {args.revision}",
            "format": args.battle_format,
            "outcome": outcome,
            "total_reward": round(total_reward, 4),
            "total_turns": step_idx,
            "turns": turns,
        }

        with open(out_path, "w") as f:
            json.dump(battle_log, f, indent=2)
        print(f"\nBattle log saved to {out_path}")
        print(f"Outcome: {outcome} | Total reward: {total_reward:.3f} | Turns: {step_idx}")

    finally:
        if showdown_proc is not None and showdown_proc.poll() is None:
            showdown_proc.terminate()


if __name__ == "__main__":
    main()
