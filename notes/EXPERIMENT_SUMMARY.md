# Experiment Summary: Quick Reference

## One-Sentence Summary
Investigating whether Llama 3.1 405B can maintain performance when learning from few-shot examples labeled by Llama 3.1 8B, measured via Performance Gap Recovered (PGR).

## Core Design (5 Key Decisions)

1. **Models**: 8B (weak) → 405B (strong) - same family, 50x parameter gap
2. **Dataset**: TruthfulQA MC (200 test, 484 train) - truthfulness focus, clear evaluation
3. **Format**: Letter-only answers (no reasoning) - simple baseline first
4. **K-Sweep**: {0, 2, 5, 10, 20} - budget-aware, covers key regimes
5. **Decoding**: Temperature=0 - deterministic, reproducible

## PGR Formula

```
PGR = (Strong_weak - Weak_baseline) / (Strong_gold - Weak_baseline)
```

- **1.0** = Perfect recovery
- **0.5** = Halfway recovery
- **0.0** = No benefit
- **<0** = Harmful

## Experimental Conditions (per K value)

| Condition | Model | Few-shot Labels | Purpose |
|-----------|-------|-----------------|---------|
| Weak baseline | 8B | None (K=0) | Lower bound |
| Strong + Gold | 405B | Gold truth | Upper bound (ceiling) |
| Strong + Weak | 405B | From 8B | Main experiment |

## Mini Experiment Results (K={0,2} on 20 questions)

```
K=0: Weak=55%, Strong=90%, PGR=1.0
K=2: Weak=55%, Strong+Gold=95%, Strong+Weak=95%, PGR=1.0
```

**Finding**: Strong model achieved perfect recovery (unaffected by weak labels)

## Why These Choices?

**Why letter-only (no reasoning)?**
- Establish simple baseline first
- Control for reasoning quality differences
- Lower cost, easier parsing
- CoT reserved as extension

**Why K={0,2,5,10,20}?**
- K=0: Baseline
- K=2: Minimal signal
- K=5,10: Standard few-shot regimes
- K=20: Test saturation
- Budget constraint prevents K>20

**Why temperature=0?**
- Reproducible (same input → same output)
- Tests model's best guess
- No sampling variance to explain
- Fair comparison across models

**Why TruthfulQA?**
- Truthfulness aligns with alignment research
- Multiple choice = reliable scoring
- Challenging enough to show gaps
- Standard benchmark

## Expected Budget

- 20 test questions: ~$2-5 (mini experiment ✓)
- 200 test questions × 5 K values: ~$50-100 (estimated)
- Buffer for extensions: ~$100
- Total: <$200 of $600 budget

## Hypotheses

**H1**: PGR will be high (>0.7) - strong models robust to weak labels
**H2**: PGR increases with K - more examples = stronger signal
**H3**: Performance gap exists - 8B: 40-60%, 405B: 80-95%

## Possible Extensions (if time/budget)

1. **Weak + Disclaimer**: Warn 405B that labels may be wrong
2. **Chain-of-Thought**: Include reasoning in demonstrations
3. **Additional pairs**: 8B→70B, 70B→405B
4. **Error analysis**: Which question types benefit most?
5. **Larger K**: Test K=30, 50 if budget allows

## Success Criteria

**Minimum**: K={0,5,10} on 200 questions, PGR calculated, plot generated
**Target**: K={0,2,5,10,20} on 200 questions, full analysis, visualizations
**Stretch**: Extensions (disclaimer/CoT/additional pairs)

## Files Generated

```
src/
├── config.py           # Environment/config management
├── model_evaluator.py  # API calls, answer extraction
├── experiment_runner.py # Orchestration, K-sweep
└── results_analyzer.py # PGR calculation, plots

test_setup.py           # Validation script
run_mini_experiment.py  # Quick test (20 questions)
DESIGN.md              # Full design decisions
EXPERIMENT_SUMMARY.md  # This file
```

## Next Steps

1. Review design decisions (DONE)
2. Run full experiment in notebook
3. Analyze results
4. Prepare presentation (slides or annotated notebook)

## Presentation Outline

**3-minute version:**
1. Problem: Can strong models learn from weak labels?
2. Approach: 8B→405B on TruthfulQA, K-sweep, measure PGR
3. Results: [Show plot, report PGR values]
4. Interpretation: [What does PGR tell us?]
5. Next steps: [Extensions based on findings]

**10-minute version:**
- Add: Design decisions rationale
- Add: Error analysis deep-dive
- Add: Comparison across K values
- Add: Discussion of implications

**30-minute version:**
- Full methodology walkthrough
- Detailed results by K
- Error case studies
- Multiple extension proposals
- Q&A preparation
