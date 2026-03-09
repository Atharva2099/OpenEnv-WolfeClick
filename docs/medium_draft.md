# Medium Article Draft: Teaching an LLM to Play Pokemon with Reinforcement Learning

## Title Options
- "I Trained an LLM to Play Competitive Pokemon — Here's What I Learned"
- "From Rock-Paper-Scissors to Pokemon: Training LLMs with GRPO"
- "Building an RL Environment for LLM Pokemon Battles"

---

## Outline

### 1. Hook (2-3 paragraphs)
- Start with the core insight: competitive Pokemon is a reasoning problem with hidden information, constrained actions, and long-term tradeoffs
- Rock-paper-scissors shows that even simple cyclic matchups create nontrivial reasoning — Pokemon scales that into a much richer domain
- The goal: build an OpenEnv-compatible environment that lets you train any LLM to play Pokemon battles using reinforcement learning

### 2. Why Pokemon? (2-3 paragraphs)
- Hidden information: you don't know your opponent's full team, movesets, or items
- Legal action constraints: only 4 moves + 5 switches per turn, must be valid
- Long-term resource management: HP, PP, team composition matter across the battle
- Active opponent: the other player adapts, creating non-stationary dynamics
- Compare to other RL benchmarks (Atari, board games) — Pokemon sits in an interesting middle ground

### 3. Environment Design (main technical section)

#### State Representation
- Markdown-formatted state with 3 sections (Part A/B/C)
- Why markdown: LLMs already understand structured text
- What information is included and why (active field, roster, opponent history)
- Show an example state snippet

#### Action Space
- JSON schema: `{"action": "move"|"switch", "choice": "name"}`
- Why constrained JSON instead of free text
- Action validation with case-insensitive, space-normalized matching
- What happens when the model hallucinates (fallback + penalty)

#### Reward Shaping
- The challenge: Pokemon battles are long (10-30 turns), sparse win/loss signal isn't enough
- Multi-component shaped reward:
  - Damage dealt/taken
  - Knockouts (+3.0/-3.0)
  - Healing (capped to prevent exploitation)
  - Setup moves (capped per Pokemon)
  - Type effectiveness bonus/penalty
  - Illegal action penalty (-10.0)
  - Anti-stall step penalty
- Design philosophy: dense signal without turning it into a toy proxy

### 4. Training Pipeline (medium-length section)

#### The Two-Stage Approach
- Stage 1: JSON warm-up SFT — teach the model to output valid action JSON
- Stage 2: GRPO — optimize the policy using real rollout data

#### Why GRPO?
- Brief explanation of Group Relative Policy Optimization
- How it differs from PPO / DPO for this use case
- The rollout collection loop: play battles, record (state, action, reward) tuples

#### Infrastructure
- Local Pokemon Showdown server via poke-env
- Colab GPU runtime for model inference
- LoRA adapters for parameter efficiency
- Multiple training runs with iterative improvement

### 5. Results & Observations (2-3 paragraphs)
- What the trained model learned to do well
- Where it still struggles
- Interesting emergent behaviors (if any)
- Comparison across checkpoints (run1 → run2 → run3)
- Honest about limitations: small training budget, random opponent, Gen 4 format

### 6. The OpenEnv Integration (1-2 paragraphs)
- What OpenEnv is and why it matters
- How the environment is packaged as a reusable server
- Link to the HF Space demo

### 7. Takeaways (2-3 paragraphs)
- What worked: structured state format, shaped rewards, GRPO on real rollouts
- What was harder than expected: battle lifecycle management, async poke-env integration, reward design
- What I'd do differently: more training budget, better opponent (self-play), broader format coverage
- The bigger picture: LLMs as RL agents in complex interactive environments

### 8. Links & Resources
- GitHub repo
- HF model weights
- HF Space demo
- OpenEnv project

---

## Key Diagrams to Include
1. Architecture diagram (Pokemon Showdown → poke-env → PokemonShowdownEnv → OpenEnv Server)
2. Training pipeline diagram (Base Model → SFT → Rollouts → GRPO → LoRA)
3. Example battle state screenshot from the HF Space
4. Reward component breakdown chart

## Estimated Length
- 1500-2000 words
- 4-5 code snippets
- 2-3 diagrams/screenshots

## Tone
- Technical but accessible
- First-person, honest about the hackathon context
- Focus on design decisions and lessons learned, not just "here's what I built"
