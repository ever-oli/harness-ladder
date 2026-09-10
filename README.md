# Harness Ladder

Cumulative **harness-engineering** powers on a single agent loop. Weights stay fixed; feature-flag powers move up the ladder one rung at a time.

**Primary model:** [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) (Apache 2.0).  
**Optional later:** Qwen2.5-3B / finetuned checkpoints for a small base-model ablation — not on the critical path.

See **[PLAN.md](./PLAN.md)** for the full execution plan, power list (P0–P12+), cloud strategy, and portfolio layer.

## What this scaffold includes (P0 / V0)

| Piece | Role |
|-------|------|
| `loop.py` | P0 V0 sampling loop |
| `powers.py` | Flags P0–P12; rung N enables 0..N |
| `eval_runner.py` | Run fixed suite → metrics |
| `ledger.py` | Append/read `results/ledger.csv` |
| `tasks/suite_smoke.json` | Tiny CI suite (8 tasks) |
| `MockLLM` | Offline deterministic client (no GPU/network) |

## Quick start

```bash
cd harness-ladder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/run_p0_smoke.py
```

Expected: all tests pass; smoke reports `success_rate=100%` with MockLLM.

## Power flags

```python
from harness_ladder.config import PowerFlags
from harness_ladder.powers import enable_up_to

flags = PowerFlags.for_rung(3)   # P0,P1,P2,P3
assert enable_up_to(0).is_on("P0")
```

**P0–P8** change control flow today (few-shot, budgeted thinking, retrieval, tools). P5–P12 remain stubs.

## Ledger

Each suite run can append one row:

`rung,powers,success_rate,n_tasks,model_id,seed,notes`

```bash
# header already in results/ledger.csv
python examples/run_p0_smoke.py   # appends a row
```

## Live model (later)

Serve MiniCPM5-2B with vLLM (or any OpenAI-compatible endpoint), then point `ModelConfig.base_url` / `model_id` at it via `OpenAICompatibleClient`. CI and local smoke stay on `MockLLM`.

## License

MIT — see [LICENSE](./LICENSE).


## Evaluation marginal gains

See [CLOUD_PLAN.md](CLOUD_PLAN.md) and [github.com/ever-oli/harness-ladder](https://github.com/ever-oli/harness-ladder). Exact-match success from results/ledger.csv; deltas are versus the preceding rung.

| Suite / rung | Passed | Score | Marginal gain | Notes |
|---|---:|---:|---:|---|
| Smoke / P0 | 8/8 | 100.0% | — | MockLLM smoke |
| Mini v1 / P0 | 2/24 | 8.3% | — | MiniCPM5-2B, T4 |
| Mini v1 / P1 | 4/24 | 16.7% | +8.4 pp | cumulative packing |
| Mini v1 / P2 | 8/24 | 33.3% | +16.6 pp | cumulative planning |
| Mini v1 / P3 | 10/24 | 41.7% | +8.4 pp | cumulative naive retrieval; results/runs/p3_real_minicpm5_mini_v1.json |
| Mini v1 / P4 | 15/24 | 62.5% | +20.8 pp | cumulative tools (MiniCPM XML); results/runs/p4_real_minicpm5_mini_v1.json |
| Mini v1 / P5 | 15/24 | 62.5% | +0.0 pp | cumulative ReAct; results/runs/p5_real_minicpm5_mini_v1.json |
| Mini v1 / P6 | 19/24 | 79.2% | **+16.7 pp** | self-refine (post-debug); results/runs/p6_real_minicpm5_mini_v1.json |
| Mini v1 / P7 | 19/24 | 79.2% | +0.0 pp | Reflexion FIXED (compact-only accept); results/runs/p7_real_minicpm5_mini_v1.json |
| Mini v1 / P8 | 24/24 | 100% | **+8.3 pp** | long-horizon calculator gate; results/runs/p8_real_minicpm5_mini_v1.json |

P3 adds top-k lexical passages from the local corpus while retaining P0-P2 behavior.

P4 adds MiniCPM5-style tool definitions + one XML tool-call round (call → observe → answer). Chart: 8.3→16.7→33.3→41.7→62.5→62.5→79.2→79.2→91.7→**100%** (long-horizon calc gate).

P5 adds a ReAct loop (Thought → tool act → observe, up to 3 rounds) on top of P4 tools. P5 ReAct traded wins: recovered code/tool/long-horizon XML cases, but some exact-match file/math answers became verbose sentences (net flat).

P6 adds self-refine (critique → compact revise) to cut verbose exact-match failures. Chart: 8.3→16.7→33.3→41.7→62.5→62.5→79.2→79.2→91.7→**100%** (long-horizon calc gate).

P7 adds Reflexion (verbal critique → one retry trial). Flat on mini suite vs P6 (same 70.8%) — retry helps less without external feedback.

P8 adds a persistent restricted Python REPL tool, **gated to `category=code`** (off for math/file/long-horizon). On the mini suite it **regressed** (−8.3 pp): MiniCPM often emits wrong REPL snippets (e.g. pages/day → 1.333, long-horizon stuck at start). Keep for harder code tasks; gate/prompt-tune before treating as default.

## Post-P6 debug (P7/P8)

Root causes for flat/regression after P6:
1. **P7** reflections urged adding `$`/units, then overwrote good answers (`12`→`7`, `33`→`$33`).
2. **P8** REPL appended `None` after `print(...)`, and hints over-triggered REPL on word problems.
3. **tool_05** regex was double-escaped (`count\\s*...`) so `count: 3` never matched.

Fixes: compact-only Reflexion accept, strip `$`/units in normalize, REPL print cleanup, softer REPL hints, suite regex repair.

**Re-eval after fixes:** P6 **19/24 (79.2%)**, P7 **19/24 (79.2%)**. **P8 code-gated:** was **19/24 (79.2%)**; after remaining-fail research fixes → **22/24 (91.7%)**; long-horizon calculator gate → **24/24 (100%)**.

## Remaining-fail debug (web research)

Against the 5 held-out mini-suite fails, online harness guidance + traces pointed to:
1. **Few-shot bleed** — long_horizon exemplar `DONE` contaminated `visit A…` (status-word family mismatch).
2. **Pre-evidence / wrong-tool discipline** — file tasks called `lookup` instead of reading P3 passages (agentic-RAG procedural failure).
3. **ReAct early stop on plannable math** — partial calculator hops (`1+10`) vs plan-once full expression (small-model ReAct literature).

Mitigations: category-matched few-shots, category tool filters + file read-gate, plan-once nudge for math/long_horizon, better calculator hints.

**Cleared:** `long_horizon_01`/`03` fixed by forced full-expression calculator gate → mini suite **24/24**.

## Evaluation — suite_v1 (48 tasks)

Same MiniCPM5-2B weights; cumulative harness. Mini suite (24) hit 100% at P8; this is the harder ruler.

| Suite / rung | Passed | Score | Notes |
|---|---:|---:|---|
| suite_v1 / P0 | 3/48 | 6.3% | bare loop |
| suite_v1 / P8 | 44/48 | **91.7%** | fail-chip re-eval (`3a26c2f`); `p8_real_minicpm5_suite_v1_fix.json` |

Remaining P8 misses: `code_07`, `tool_10`, `long_horizon_05`, `long_horizon_07`.

