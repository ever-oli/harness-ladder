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

**P0–P4** change control flow today (few-shot, budgeted thinking, retrieval, tools). P5–P12 remain stubs.

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
| Mini v1 / P4 | 15/24 | 62.5% | **+20.8 pp** | cumulative tools (MiniCPM XML); results/runs/p4_real_minicpm5_mini_v1.json |

P3 adds top-k lexical passages from the local corpus while retaining P0-P2 behavior.

P4 adds MiniCPM5-style tool definitions + one XML tool-call round (call → observe → answer). Chart: 8.3→16.7→33.3→41.7→**62.5**.
