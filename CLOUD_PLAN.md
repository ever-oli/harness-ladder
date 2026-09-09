# Harness Ladder — Cloud Execution Plan

**Base model:** MiniCPM5-2B (`openbmb/MiniCPM5-2B`) — 2.5B params, Apache 2.0, standard
`LlamaForCausalLM`, 131K native context, current SOTA under 4B, strong tool-use/agent
scores (97.1 τ²-Bench Telecom, 66.6 BFCL v4). fp16 weights ≈ 5GB → fits a T4 (16GB)
with ample KV-cache headroom.

**Objective:** implement the 21 DAIR.AI harness-engineering papers as one-added-power-per-paper
on a single stable agent loop. Deliverable: ablation chart of marginal gain per power +
capstone re-run of the ladder on 3 bases (MiniCPM5-2B, Qwen2.5-3B, froglet checkpoints).

Reference: https://academy.dair.ai/papers/collections/harness-engineering

**Cloud default:** Lightning AI Studios for development and evals. Kaggle remains optional later for extra sweep hours if needed — not required for Week 1.

---

## 1. Cloud strategy

| | Lightning AI (primary) | Kaggle (optional) |
|---|---|---|
| Free GPU | Limited free GPU hours/mo (verify current allowance) | ~30 h/week (T4×2 or P100) |
| Session limits | Persistent Studio | 9 h max, then killed |
| Persistence | Filesystem persists | Must save to Kaggle Datasets |
| Best for | **Development + primary evals** | Extra overnight sweeps |

One repo, one `requirements.txt`. Develop and run on Lightning first.

## 2. Model serving on a T4

- **Engine:** vLLM (`vllm serve openbmb/MiniCPM5-2B`) for throughput; fall back to
  Hugging Face `transformers` fp16 for simplicity/debugging.
- **Context cap:** 8–32K for normal harness iterations (speed). Reserve the full 131K
  for the long-file / long-horizon powers (P12, P17) where it matters.
- **Quantization:** optional. fp16 fits fine; reach for 4-bit (AWQ/GPTQ) only if
  agentic-loop latency becomes the bottleneck.
- **Latency discipline:** agentic loops are sequential-call-bound, not throughput-bound.
  Keep `max_tokens` small per call, batch independent calls, prefer vLLM's continuous
  batching during sweeps.

## 3. The ruler: fixed task suite

Same ~100 tasks at every rung. Suggested mix:
- Code repair / synthesis (MBPP-style, ~30)
- Math word problems (GSM8K-style, ~20)
- Tool-use QA (multi-hop, needs calls, ~20)
- File-grounded QA over a synthetic repo (retrieval/recursive-ask powers, ~15)
- Long-horizon multi-step tasks (sub-agent/memory powers, ~15)

Per-power metrics: **success rate, avg tokens, avg wall time**. Marginal gain = this
rung minus previous rung. That delta is the chart.

## 4. The 21 rungs (one power per paper, cumulative)

- **P0** V0 sampling loop (baseline — everything measured against this)
- **P1** Few-shot examples in prompt
- **P2** Budgeted thinking (fixed reasoning token budget)
- **P3** Retrieval (naive RAG over task corpus)
- **P4** Tool definitions + calling
- **P5** ReAct-style loop (thought → act → observe)
- **P6** Self-refine (generate → critique → revise)
- **P7** Reflexion (verbal reinforcement across trials)
- **P8** Persistent Python REPL tool
- **P9** Sub-agents (spawn with scoped context)
- **P10** Verified skills (pre-tested tool/skill library)
- **P11** Memory CRUD (persistent memory across tasks)
- **P12** Recursive asks over large files (131K context power)
- **P13** DSPy-style prompt/program optimization
- **P14** GEPA-lite trace evolution
- **P15** Prime-style preset (distilled best-config)
- **P16** OpenJarvis-lite (OSWorld-style computer-use tasks)
- **P17** METR-lite horizon measurement (time-to-50%-success per power)
- **P18** Human-gated self-modification (prompts/skills only, never weights)
- **P19–P20** Remaining collection entries — map to papers when scaffolding
  (collection lists 21 entries; confirm final two against the page)

Each power = one feature flag, default off. Rung N = flags 0..N on.

## 5. Run protocol (per rung)

1. Implement the power behind its flag; unit-test the flag in isolation.
2. Run the full task suite with flags 0..N on. Fixed seeds.
3. Append metrics to the results ledger (`results/ledger.csv` — versioned, synced off
   the ephemeral machine after every rung).
4. Sanity-check: if a power shows ~zero gain, spot-ablate (that power alone on V0)
   before concluding it's useless vs. the base being too weak to exploit it.
5. Git-tag each rung (`rung-p07-reflexion`, …).

## 6. Persistence & reproducibility

- Lightning: push to GitHub at end of every work session; ledger lives in-repo under `results/`.
- Pin everything: model revision, `requirements.txt`, seeds, vLLM version.
- Optional Kaggle: chunk sweeps ≤9 h; checkpoint ledger to a Dataset after every rung.

## 7. Timeline (rough, ~5 weeks)

- **Week 1:** Lightning env + serving + task suite + P0 loop + eval runner. First chart point.
- **Weeks 2–4:** 5–7 powers/week, each with a full suite run. Watch the chart grow.
- **Week 5:** capstone — re-run ladder on Qwen2.5-3B and froglet checkpoints; final
  3-base comparison chart; README + recorded demo.

## 8. Risks & mitigations

- **Agentic latency on T4** → small `max_tokens`, vLLM, constrained/structured decoding
  for tool calls (small models make tool-call parsing noisy).
- **Weak-signal powers** → P12 131K-context and tool-use should shine on MiniCPM5-2B;
  flat reasoning powers are still a writeup finding.
- **Free-tier quota** → track GPU-hours per rung in the ledger; schedule heavy P13/P14
  carefully.
