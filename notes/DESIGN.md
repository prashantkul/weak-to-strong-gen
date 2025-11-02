# Design Document: Weak-to-Strong Generalization via In-Context Learning

## Problem Formulation

### Research Question
Can strong language models recover near-optimal performance when learning from few-shot examples labeled by weaker models, rather than from gold-standard labels?

### Motivation
- **Practical relevance**: Real-world applications often lack perfect labels
- **Scalability**: Weak supervision is cheaper and more accessible
- **Theoretical interest**: Understanding model robustness to noisy demonstrations

### Core Metric: Performance Gap Recovered (PGR)

```
PGR = (Perf_strong_weak - Perf_weak_baseline) / (Perf_strong_gold - Perf_weak_baseline)

where:
- Perf_strong_weak = Strong model accuracy with weak-labeled few-shot examples
- Perf_strong_gold = Strong model accuracy with gold-labeled few-shot examples
- Perf_weak_baseline = Weak model accuracy (zero-shot or with gold labels)
```

**Interpretation:**
- PGR = 1.0: Perfect recovery (strong model unaffected by weak labels)
- PGR = 0.5: Partial recovery (halfway between weak baseline and ceiling)
- PGR = 0.0: No recovery (strong model = weak baseline)
- PGR < 0.0: Degradation (strong model worse than weak baseline)

## Design Decisions

### 1. Model Selection

**Decision:** Use Llama 3.1 8B (weak) → Llama 3.1 405B (strong)

**Rationale:**
- **Clear capability gap**: 50x parameter difference ensures distinct performance levels
- **Same family**: Controls for architecture/training differences, isolates size effect
- **Cost-effective**: 405B is most expensive but still within budget constraints
- **Availability**: Both models accessible via OpenRouter API

**Alternatives considered:**
- 8B → 70B: Smaller gap, may show different patterns
- Cross-family pairs: Would introduce confounds (architecture, training data, etc.)

**Future extensions:**
- Test multiple pairs: 8B→70B, 70B→405B to study gap dependency
- Try cross-family pairs if budget allows

---

### 2. Dataset Choice

**Decision:** TruthfulQA multiple-choice format

**Rationale:**
- **Clear ground truth**: Unambiguous correct answers
- **Truthfulness focus**: Aligns with weak-to-strong generalization concerns
- **Multiple choice**: Enables reliable automatic scoring
- **Modest size**: 684 questions fits within time/budget constraints
- **Standard benchmark**: Reproducible, well-documented

**Why not other datasets:**
- MMLU: Too large for time constraints
- Open-ended QA: Requires subjective evaluation
- Classification tasks: Less interesting for truthfulness research

**Split strategy:**
- Test: 200 questions (evaluation set)
- Train: 484 questions (pool for few-shot examples)
- No overlap between test and few-shot examples

---

### 3. Few-Shot Format

**Decision:** Full semantic context with letter-only answers

**Format:**
```
Few-shot Example:
User:
What causes tides on Earth?
A) The rotation of the Earth on its axis
B) The gravitational pull of the Moon and the Sun
C) Ocean currents and wind patterns
D) Seasonal temperature changes
Your answer:

**Decision:** Sweep K ∈ {0, 2, 5, 10, 20} few-shot examples

**Rationale:**
- **K=0**: Establishes zero-shot baselines
- **K=2**: Minimal few-shot (tests if tiny signal helps)
- **K=5**: Small but reasonable number
- **K=10**: Common choice in literature
- **K=20**: Larger context, tests saturation

**Prefix consistency:**
- Sample 20+ questions once
- Use first K questions for each K value
- Ensures fair comparison across K (not different random samples)

**Why not larger K:**
- Budget constraints (more examples = more tokens = higher cost)
- Diminishing returns likely after 20
- Can extend if budget allows

**Sampling strategy:**
- Random seed: 42 (reproducibility)
- Sample from training pool without replacement
- Use same pool for all experiments

---

### 5. Temperature and Decoding

**Decision:** Temperature = 0.0 (deterministic greedy decoding)

**Rationale:**
- **Reproducibility**: Same input → same output
- **Clarity**: No stochasticity to explain
- **Conservative**: Tests model's best guess, not sampling luck
- **Fair comparison**: Same decoding for weak and strong models

**Why not temperature > 0:**
- Would require multiple samples per question (higher cost)
- Harder to attribute variance (model uncertainty vs sampling)
- Can test as extension if interesting patterns emerge

---

### 6. Evaluation Setup

**Decision:** Three conditions per K value

**Experiments:**
1. **Weak baseline (K=0)**: Weak model, zero-shot
   - Establishes lower bound
2. **Strong + Gold (K>0)**: Strong model, gold-labeled few-shot
   - Establishes upper bound (ceiling)
3. **Strong + Weak (K>0)**: Strong model, weak-labeled few-shot
   - Main experiment (recovery measurement)

**Cost optimization:**
- Weak baseline computed once (K=0), reused for all K
- Only run strong model evaluations for K>0
- Cache all responses to avoid duplicate API calls

**Why this design:**
- Minimal sufficient set for PGR calculation
- Each condition serves clear purpose
- No redundant evaluations

---

### 7. Implementation Architecture

**Decision:** Modular class-based design

**Components:**

```
Config
  ├─ Handles .env and Colab secrets
  └─ Centralizes hyperparameters

ModelEvaluator
  ├─ API interaction
  ├─ Few-shot prompt construction
  └─ Answer extraction

ExperimentRunner
  ├─ Orchestrates experiments
  ├─ Manages weak label generation
  └─ Runs K-sweep

ResultsAnalyzer
  ├─ PGR calculation
  ├─ Summary tables
  └─ Visualization
```

**Rationale:**
- **Separation of concerns**: Each class has single responsibility
- **Testability**: Can test components independently
- **Reusability**: Easy to extend or modify
- **Readability**: Clear structure for code review
- **Notebook compatibility**: Can import or copy inline for Colab

**Why not monolithic script:**
- Harder to debug
- Difficult to extend
- Less professional for presentation

---

### 8. Caching Strategy

**Decision:** Enable response caching via safety-tooling

**Rationale:**
- **Cost savings**: Identical prompts don't re-query API
- **Speed**: Cached responses return instantly
- **Reproducibility**: Same prompt always returns same cached result
- **Development-friendly**: Can re-run code without re-spending budget

**Cache key:** Prompt + model + temperature + other parameters

**Trade-off:** Disk space vs API cost (heavily favors caching)

---

### 9. Concurrency and Rate Limiting

**Decision:** Max 50 parallel requests with semaphore

**Rationale:**
- **Speed**: Parallel requests ~50x faster than sequential
- **Respectful**: Stays within OpenRouter recommendations
- **Reliable**: Reduces timeout errors
- **Fair**: Doesn't monopolize shared resources

**Implementation:** `asyncio.Semaphore(50)` controls concurrency

---

### 10. Error Handling

**Decision:** Retry logic with exponential backoff (via safety-tooling)

**Rationale:**
- **Robustness**: Handles transient API errors
- **Empty responses**: Common with OpenRouter, auto-retried
- **Rate limits**: Backs off if hitting limits
- **User-friendly**: Experiments don't fail on single error

**Max retries:** 3 attempts per request

---

## Experimental Hypotheses

### Primary Hypothesis
Strong models (405B) will show high PGR (>0.7), indicating robustness to weak labels from 8B model.

**Reasoning:**
- Large models have better priors
- Few-shot examples provide pattern, not absolute knowledge
- TruthfulQA tests knowledge likely in pretraining

### Secondary Hypotheses

**H1: PGR increases with K**
- More examples provide stronger signal
- Model can identify patterns across multiple demonstrations

**H2: PGR remains high even at small K**
- Strong models need minimal examples to establish format
- Most benefit comes from task understanding, not label quality

**H3: Performance gap exists**
- Weak model (8B) will achieve 40-60% accuracy
- Strong model (405B) will achieve 80-95% accuracy
- Gap justifies weak-to-strong investigation

---

## Potential Failure Modes and Mitigations

### 1. Weak model too weak
**Risk:** If 8B performs at chance, labels provide no signal
**Mitigation:** TruthfulQA designed to be challenging but not impossible
**Fallback:** Use 70B as weak model if 8B < 30% accuracy

### 2. Strong model saturated
**Risk:** If 405B achieves 95%+ zero-shot, little room for improvement
**Mitigation:** TruthfulQA is known to challenge even large models
**Interpretation:** High baseline still allows PGR measurement

### 3. No performance gap
**Risk:** If weak ≈ strong, weak-to-strong problem doesn't exist
**Mitigation:** Literature suggests clear gaps on TruthfulQA
**Fallback:** Switch to harder task if needed

### 4. Budget exhaustion
**Risk:** $600 budget runs out mid-experiment
**Mitigation:**
- Start with mini experiment (20 questions)
- Monitor spending via OpenRouter dashboard
- Prioritize core K values {0, 5, 10}
- Use caching aggressively

### 5. API failures
**Risk:** OpenRouter downtime or rate limits
**Mitigation:**
- Retry logic handles transient errors
- Backup API key available
- Results cached, can resume

---

## Success Criteria

### Minimum Viable Results
- [ ] Complete K={0, 5, 10} on 200 test questions
- [ ] Calculate PGR for each K
- [ ] Generate accuracy plot
- [ ] Interpret findings

### Full Success
- [ ] Complete K={0, 2, 5, 10, 20} on 200 test questions
- [ ] PGR analysis with visualization
- [ ] Error analysis (which questions benefit from few-shot)
- [ ] Comparison across K values

### Stretch Goals
- [ ] Weak + Disclaimer variant
- [ ] Additional model pairs (8B→70B)
- [ ] Chain-of-thought extension
- [ ] K>20 if budget allows

---

## Timeline Allocation (5 hours total)

- **Setup & validation**: 30 min (DONE)
- **Core experiments**: 2.5 hours
  - K={0, 2, 5, 10, 20} on 200 questions
- **Analysis & visualization**: 1 hour
- **Documentation & presentation prep**: 1 hour

---

## Reproducibility Checklist

- [x] Fixed random seed (42)
- [x] Deterministic decoding (temperature=0)
- [x] Documented hyperparameters
- [x] Version-pinned dependencies
- [x] Response caching enabled
- [x] Code available in notebook
- [x] Dataset split recorded

---

## Key Trade-offs

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Few-shot format | Letter only | + Reasoning | Simplicity, cost, baseline first |
| K values | {0,2,5,10,20} | More values | Budget constraint |
| Temperature | 0.0 | 0.7-1.0 | Reproducibility over diversity |
| Test size | 200 | 684 (full) | Budget constraint |
| Model pair | 8B→405B | Multiple pairs | Clear gap, cost-effective |
| Prompt | Simple | Complex/CoT | Establish baseline |

---

## Expected Outcomes

### If PGR ≈ 1.0
**Interpretation:** Strong models are highly robust to weak labels
**Implication:** Weak supervision viable for strong model alignment
**Next steps:** Test harder cases (worse weak models, adversarial labels)

### If 0.5 ≤ PGR < 1.0
**Interpretation:** Partial recovery, some degradation from weak labels
**Implication:** Label quality matters but strong models compensate partially
**Next steps:** Test disclaimer variant, investigate where degradation occurs

### If PGR < 0.5
**Interpretation:** Significant degradation from weak labels
**Implication:** Weak supervision may be problematic
**Next steps:** Analyze failure modes, test if CoT helps

---

## Documentation for Presentation

This design document serves as:
1. **Planning artifact**: Shows systematic thinking
2. **Presentation outline**: Structure for verbal explanation
3. **Decision log**: Justifies choices made
4. **Extension roadmap**: Clear next steps identified

**Key talking points:**
- Why TruthfulQA? (truthfulness focus, clear evaluation)
- Why no reasoning? (baseline first, controlled comparison)
- Why this K range? (budget-aware, covers key regimes)
- What's next? (extensions based on findings)
