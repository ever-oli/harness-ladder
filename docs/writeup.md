# Harness Ladder: what agent papers buy on a frozen 2B

**Ever Olivares** · harness engineering portfolio note · Sep 2026

Weights stay fixed. The harness moves. That is the whole bet.

This project stacks DAIR/YC-style agent powers — few-shot, budgeted thinking, RAG, tools, ReAct, self-refine, Reflexion, a gated Python REPL — as **cumulative flags** on one loop, and measures exact-match success on the same task suites after each rung. The base model is always [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B), run fp16 on free Lightning AI T4s. No finetune. No bigger teacher. Just scaffolding.

Code: [ever-oli/harness-ladder](https://github.com/ever-oli/harness-ladder) · Charts: [docs/portfolio](./portfolio/index.html)

## Why this framing

Most “agent demos” change three variables at once: model, prompts, and tools. You cannot tell which paper moved the needle. A **harness ladder** freezes the weights and the suite, then adds one power per rung. The chart is marginal gain: rung \(N\) minus rung \(N-1\).

That matches how practitioners actually ship small models on free compute: you rarely get another billion parameters, but you can afford another retry loop.

## Method (short)

- **Loop:** shared `run_v0_loop` with feature flags `P0…P8`.
- **Serving:** Hugging Face Transformers on T4 (fp16); chat template from MiniCPM.
- **Grading:** exact match after light deterministic normalize (units, list spacing, etc.).
- **Ledger:** every run appends `results/ledger.csv` and a JSON trajectory dump under `results/runs/`.
- **Suites:** fixed JSON task lists so re-runs are comparable.

Powers in one line: P0 bare loop → P1 few-shot → P2 budgeted thinking → P3 lexical RAG → P4 MiniCPM XML tools → P5 ReAct → P6 self-refine → P7 Reflexion → P8 gated `python_repl`.

## Result 1 — suite_v1 saturates

`tasks/suite_v1.json` is 48 custom exact-match items (code, math, tools, file-grounded, long-horizon). It is deliberately compact so a 2B can show structure without drowning in format noise.

| Rung | Score |
|------|------:|
| P0 | 6.3% |
| P1 | 68.8% |
| P2 | 72.9% |
| P3 | 75.0% |
| P4 | 75.0% |
| P5 | 85.4% |
| P6 | 95.8% |
| P7 | **100%** |
| P8 | **100%** |

**Takeaways**

1. **Few-shot is the first cliff** (+62.5 pp). Without exemplars, MiniCPM narrates; with them, it emits bare answers.
2. **Tools alone are not the story here.** P4 matches P3. On this suite, XML tools without ReAct discipline barely help.
3. **Self-refine and Reflexion close the last gap.** P5→P6 (+10.4 pp) and P6→P7 (+4.2 pp) finish the climb. P8 holds the perfect score but is not required for saturation.

A mini 24-task subset also hit 100% at P8 earlier; suite_v1 was the first “harder ruler” that still eventually flatlined.

## Result 2 — suite_v2 restores headroom

After 100%, the ladder stops teaching. **suite_v2** samples a fixed seed (`42`) mix from papers:

| Slice | n | Source |
|-------|--:|--------|
| Math | 24 | GSM8K (Cobbe et al., 2021) |
| Reasoning | 24 | BIG-Bench Hard (Suzgun et al., 2022) |
| Code | 16 | MBPP (Austin et al., 2021) |

Endpoints on the same MiniCPM5-2B:

| Rung | Score |
|------|------:|
| P0 | 18.8% (12/64) |
| P7 | **48.4%** (31/64) |
| P8 | 40.6% (26/64) |

Harness still helps (+29.7 pp P0→P7), but absolute performance is back in the interesting band. Mid-rungs are not recorded yet — Lightning Studio ran out of credits mid re-eval.

## Negative result — P8 REPL tax

On suite_v2, **P8 underperformed P7 by 7.8 pp**. Traces show MiniCPM emitting broken tool XML (`…]]>`) on full “Implement and run…” MBPP prompts instead of defining functions and printing results. Five tasks that P7 passed flipped to fail once the REPL was offered.

Mitigation in flight: skip P8 REPL when the prompt looks like a full MBPP implement task (keep short snippet-code REPL). Re-measurement blocked on GPU credits.

This is the useful kind of failure: a paper power that helps on toy code items can hurt when the model cannot operate the tool interface.

## What I am not claiming

- Not SOTA on GSM8K/BBH/MBPP. A 2B on exact match will look weak next to 70B CoT numbers.
- Not that Reflexion “solves” agents. It solved *this* custom suite’s remaining format/value errors.
- Not that RAG/tools are useless — only that on suite_v1’s distribution they were not the binding constraint.

## Next (when free GPU returns)

1. Finish suite_v2 mid-rungs (P2/P4/P6) and confirm the P8 MBPP gate.
2. Marimo ledger/trace explorer (museum + playground) as planned in `PLAN.md`.
3. Optional second base (Qwen2.5-3B) as an ablation — same harness, different weights.

## Repro

```bash
git clone https://github.com/ever-oli/harness-ladder
cd harness-ladder
# suite_v2 rebuild (optional): python scripts/build_suite_v2.py
PYTHONPATH=src python examples/run_p4_transformers.py \
  --rung 7 --suite tasks/suite_v1.json \
  --output results/runs/p7_real_minicpm5_suite_v1.json
```

Open `docs/portfolio/index.html` locally for charts (needs the adjacent `chart-data.json`).

---

*Built on free T4 quotas. If the only thing this project proves is that scaffolding still matters when you cannot buy parameters — that is enough for a portfolio.*
