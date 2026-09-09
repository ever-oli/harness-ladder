# Harness Ladder — Execution Plan

**Primary base:** `openbmb/MiniCPM5-2B` (Apache 2.0, ~2.5B, LlamaForCausalLM, long context). Weights stay fixed; harness powers move.

**Secondary (optional later):** Qwen2.5-3B / finetuned checkpoints for a 2–3 base capstone ablation — **not** on the critical path.

**Objective:** Implement DAIR/YC harness-engineering powers as one-added-power-per-rung on a single agent loop. Deliverable: marginal-gain chart + portfolio site (static + marimo results explorer).

Reference: https://academy.dair.ai/papers/collections/harness-engineering

---

## Cloud strategy (hybrid)

| | Lightning AI | Kaggle |
|--|--|--|
| Best for | Dev / IDE iteration | Long eval sweeps |
| Notes | Persistent Studio | ~30 h/week GPU; chunk ≤9 h; ledger off-box every rung |

One repo, one `requirements.txt`, runs in both.

## Serving on T4

- Prefer **vLLM** (`vllm serve openbmb/MiniCPM5-2B`); HF `transformers` fp16 fallback.
- Context: 8–32K default; reserve longer context for P12+.
- Agentic loops are latency-bound: small `max_tokens`, structured tool calls.

## Ruler: fixed task suite (~100)

Same tasks every rung: code (~30), math (~20), tool-use (~20), file-grounded (~15), long-horizon (~15).

Metrics: success rate, avg tokens, avg wall time. **Marginal gain = rung N − rung N−1.**

## Powers (cumulative feature flags)

**Phase A — ship solid:** P0–P12 (V0 → recursive ask)  
**Phase B — lite:** P13 DSPy-style, P14 GEPA-lite  
**Phase C — capstone:** Prime-shaped preset + METR-lite horizon + human-gated skill/prompt edits  
**Docs/stubs only in v1:** full DGM / Meta-Harness / Continual weight updates / OpenJarvis computer-use

Each power = one flag, default off. Rung N = flags 0..N on.

## Run protocol

1. Implement + unit-test flag  
2. Full suite, fixed seeds  
3. Append `results/ledger.csv` (versioned; sync off ephemeral machines)  
4. Spot-ablate zero-gain powers  
5. Git tag `rung-pXX-...`

## Portfolio layer

- Static site: charts, writeups, demo video  
- Marimo: ledger/trace explorer (museum + playground)  
- **Not** WebGPU hosting of the live 2B harness

## Timeline (~5 weeks)

- W1: envs + serving + suite + P0 + eval runner  
- W2–4: 5–7 powers/week with full suite runs  
- W5: Prime preset + optional Qwen ablation + site/marimo + README demo

## Language

Persistent **Python REPL** default. No Julia required.


See also [CLOUD_PLAN.md](CLOUD_PLAN.md) (canonical cloud execution plan).
