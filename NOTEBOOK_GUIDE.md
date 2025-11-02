# Jupyter Notebook Execution Guide

Complete guide for running weak-to-strong generalization experiments in a Jupyter notebook.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Baseline Experiments](#baseline-experiments)
4. [Disclaimer Experiments](#disclaimer-experiments)
5. [Chain-of-Thought Preparation](#chain-of-thought-preparation)
6. [Chain-of-Thought Experiments](#chain-of-thought-experiments)
7. [Visualization](#visualization)

---

## Prerequisites

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install python-dotenv datasets pandas matplotlib
!pip install "safetytooling @ git+https://github.com/safety-research/safety-tooling.git@unpinned_requirements"
```

### Cell 2: Setup Imports

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd()
if (project_root / "src").exists():
    sys.path.insert(0, str(project_root))
    print(f"✓ Added {project_root} to sys.path")

# Import our experiment functions
from notebook_experiments import (
    run_baseline_sweep,
    run_disclaimer_sweep,
    run_cot_sweep
)

print("✓ Imports complete")
```

### Cell 3: Setup Environment

```python
from src.config import Config

# Load configuration from .env file
config = Config.from_env()
config.setup_environment()

print(f"✓ Weak Model:  {config.weak_model}")
print(f"✓ Strong Model: {config.strong_model}")
print(f"✓ Temperature: {config.temperature}")
print(f"✓ Cache directory: {config.cache_dir}")
```

---

## Dataset Preparation

### Cell 4: Load and Inspect Dataset

```python
from src import DatasetManager

dm = DatasetManager()
test_data, few_shot_pool, split = dm.load_split()

print(f"✓ Test set size: {len(test_data)}")
print(f"✓ Few-shot pool size: {len(few_shot_pool)}")
print(f"✓ Split name: {split}")

# Show example question
print("\nExample question:")
print(test_data[0].question)
print(f"Answer: {test_data[0].answer}")
```

**Expected Output:**
- Test set: 200 questions
- Few-shot pool: 617 questions
- Example question with multiple choice format

---

## Baseline Experiments

### Cell 5: Run 405B Baseline (8B→405B)

```python
# Run baseline sweep for 405B
# This will test K={0, 2, 5, 10}
# Expected time: ~15-20 minutes

results_405b = await run_baseline_sweep(
    model_pair="8b_to_405b",
    k_values=[0, 2, 5, 10],
    save_results=True
)

print("\n" + "="*70)
print("405B BASELINE COMPLETE")
print("="*70)
```

**What this does:**
1. Generates weak labels from 8B model (10 examples)
2. Runs weak baseline at K=0
3. Runs strong baseline at K=0
4. For K∈{2,5,10}: Runs strong+gold and strong+weak
5. Calculates PGR for each K
6. Saves results to `results/8b_405b_baseline_TIMESTAMP/`

**Expected Results:**
- K=0: PGR = 1.000 (100%)
- K=2: PGR ≈ 0.889 (88.9%)
- K=5: PGR = 1.000 (100%)
- K=10: PGR ≈ 0.984 (98.4%)

### Cell 6: Run 70B Baseline (8B→70B)

```python
# Run baseline sweep for 70B
# Expected time: ~10-15 minutes

results_70b = await run_baseline_sweep(
    model_pair="8b_to_70b",
    k_values=[0, 2, 5, 10],
    save_results=True
)

print("\n" + "="*70)
print("70B BASELINE COMPLETE")
print("="*70)
```

**Expected Results:**
- K=0: PGR = 1.000 (100%)
- K=2: PGR ≈ 0.920 (92.0%)
- K=5: PGR ≈ 0.930 (93.0%)
- K=10: PGR ≈ 0.864 (86.4%) ⚠️ **Degradation at high K!**

**Key Finding:** 70B shows degradation at K=10, while 405B maintains high PGR.

---

## Disclaimer Experiments

### Cell 7: Run 405B Disclaimer

```python
# Run disclaimer experiment for 405B
# Must specify path to baseline experiment for comparison
# Expected time: ~10 minutes

results_405b_disclaimer = await run_disclaimer_sweep(
    model_pair="8b_to_405b",
    k_values=[0, 2, 5, 10],
    baseline_exp_path="results/8b_405b_baseline_TIMESTAMP/experiment.json",  # Update with actual timestamp
    save_results=True
)

print("\n" + "="*70)
print("405B DISCLAIMER COMPLETE")
print("="*70)
```

**What this does:**
1. Loads baseline results for comparison
2. Runs experiments with disclaimer prompt
3. Calculates delta from baseline
4. Saves results with comparison

**Expected Results:**
- Minimal effect on 405B (±0.02 PGR change)
- 405B is already robust, disclaimer doesn't help much

### Cell 8: Run 70B Disclaimer

```python
# Run disclaimer experiment for 70B
# Expected time: ~8 minutes

results_70b_disclaimer = await run_disclaimer_sweep(
    model_pair="8b_to_70b",
    k_values=[0, 2, 5, 10],
    baseline_exp_path="results/8b_70b_baseline_TIMESTAMP/experiment.json",  # Update with actual timestamp
    save_results=True
)

print("\n" + "="*70)
print("70B DISCLAIMER COMPLETE")
print("="*70)
```

**Expected Results (K-dependent reversal):**
- K=0: +0.067 improvement
- K=2: +0.080 improvement ✓ **Best improvement!**
- K=5: +0.017 minimal
- K=10: -0.017 degradation ✗ **Hurts at high K!**

**Key Finding:** Disclaimer helps 70B at low K but hurts at high K!

---

## Chain-of-Thought Preparation

### Cell 9: Generate Weak CoT Labels

```python
# Generate CoT-enhanced weak labels from 8B model
# This creates reasoning demonstrations for few-shot examples
# Expected time: ~5 minutes

import asyncio
from src import Config, DatasetManager, ModelEvaluator
from src.model_evaluator import ModelResponse
import json
from datetime import datetime
from pathlib import Path

config = Config.from_env()
config.setup_environment()

pair = __import__('src').get_model_pair("8b_to_405b")
weak_model_id = pair.weak_model

dm = DatasetManager()
test_data, few_shot_pool, split = dm.load_split()

# Generate CoT labels for first 20 examples
evaluator = ModelEvaluator(config, use_cot=True)
questions_to_label = few_shot_pool[:20]
questions = [(q.question_id, q.question) for q in questions_to_label]

weak_cot_responses = await evaluator.evaluate_batch(
    questions=questions,
    model_id=weak_model_id,
    few_shot_prompt=None,
    verbose=True
)

# Calculate accuracy
gt_map = {q.question_id: q.answer for q in questions_to_label}
num_correct = sum(1 for r in weak_cot_responses if r.answer == gt_map[r.question_id])
accuracy = num_correct / len(weak_cot_responses)

print(f"\n8B Weak Model with CoT Accuracy: {accuracy:.2%}")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("data/cot_weak_labels")
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / f"8b_cot_weak_labels_{timestamp}.json"

data = {
    "metadata": {
        "timestamp": timestamp,
        "weak_model": weak_model_id,
        "temperature": config.temperature,
        "num_labels": len(weak_cot_responses),
        "accuracy": accuracy,
        "use_cot": True
    },
    "weak_labels": [r.model_dump() for r in weak_cot_responses]
}

with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Saved to: {output_file}")
weak_cot_path = str(output_file)
```

**Expected Output:**
- 8B with CoT accuracy: ~70%
- File saved to `data/cot_weak_labels/8b_cot_weak_labels_TIMESTAMP.json`

### Cell 10: Generate Gold CoT Labels

```python
# Generate CoT-enhanced gold labels from 405B model
# Expected time: ~10 minutes (405B is slower)

strong_model_id = pair.strong_model

gold_cot_responses = await evaluator.evaluate_batch(
    questions=questions,
    model_id=strong_model_id,
    few_shot_prompt=None,
    verbose=True
)

# Calculate accuracy
num_correct = sum(1 for r in gold_cot_responses if r.answer == gt_map[r.question_id])
accuracy = num_correct / len(gold_cot_responses)

print(f"\n405B Strong Model with CoT Accuracy: {accuracy:.2%}")

# Save
output_dir = Path("data/cot_gold_labels")
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / f"405b_cot_gold_labels_{timestamp}.json"

data = {
    "metadata": {
        "timestamp": timestamp,
        "strong_model": strong_model_id,
        "temperature": config.temperature,
        "num_labels": len(gold_cot_responses),
        "accuracy": accuracy,
        "use_cot": True
    },
    "gold_labels": [r.model_dump() for r in gold_cot_responses]
}

with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Saved to: {output_file}")
gold_cot_path = str(output_file)
```

**Expected Output:**
- 405B with CoT accuracy: ~75%
- File saved to `data/cot_gold_labels/405b_cot_gold_labels_TIMESTAMP.json`

---

## Chain-of-Thought Experiments

### Cell 11: Run 405B CoT

```python
# Run CoT experiments for 405B
# Uses the weak and gold CoT labels generated above
# Expected time: ~10 minutes

results_405b_cot = await run_cot_sweep(
    model_pair="8b_to_405b",
    k_values=[0, 2, 5, 10],
    baseline_exp_path="results/8b_405b_baseline_TIMESTAMP/experiment.json",  # Update timestamp
    weak_cot_labels_path=weak_cot_path,  # From Cell 9
    gold_cot_labels_path=gold_cot_path,  # From Cell 10
    save_results=True
)

print("\n" + "="*70)
print("405B COT COMPLETE")
print("="*70)
```

**Expected Results (Crossover Effect!):**
- K=0: -0.164 ✗ **Hurts without examples**
- K=2: +0.095 ✓ **Largest gain!**
- K=5: 0.000 (neutral)
- K=10: +0.064 ✓ **Helps at high K**

**Key Finding:** CoT shows crossover - hurts at K=0, helps at K>1!

### Cell 12: Run 70B CoT

```python
# Run CoT experiments for 70B
# Expected time: ~8 minutes

results_70b_cot = await run_cot_sweep(
    model_pair="8b_to_70b",
    k_values=[0, 2, 5, 10],
    baseline_exp_path="results/8b_70b_baseline_TIMESTAMP/experiment.json",  # Update timestamp
    weak_cot_labels_path=weak_cot_path,
    gold_cot_labels_path=gold_cot_path,
    save_results=True
)

print("\n" + "="*70)
print("70B COT COMPLETE")
print("="*70)
```

**Expected Results:**
- K=0: +0.044 (slight help)
- K=2: -0.020 ✗ **Hurts!**
- K=5: -0.035 ✗ **Hurts!**
- K=10: -0.017 ✗ **Hurts!**

**Key Finding:** CoT consistently hurts 70B at K>0 - weak reasoning confuses it!

---

## Visualization

### Cell 13: Create Final Comparison

```python
# Generate comprehensive comparison visualization
# Shows all 6 conditions: baseline/disclaimer/CoT × 405B/70B

!python create_final_comparison.py
```

**Output:**
- `results/final_comprehensive_comparison.png`
- 6-panel visualization showing:
  1. 405B: All interventions
  2. 70B: All interventions
  3. Baseline comparison (405B vs 70B)
  4. Disclaimer comparison
  5. CoT comparison
  6. Intervention effectiveness (bar chart)

### Cell 14: Display Results

```python
from IPython.display import Image, display
import pandas as pd

# Display the visualization
display(Image('results/final_comprehensive_comparison.png'))

# Create summary table
summary_data = []
for k in [0, 2, 5, 10]:
    summary_data.append({
        "K": k,
        "405B Baseline": f"{results_405b['pgr_results'][k].pgr:.3f}",
        "405B Disclaimer": f"{results_405b_disclaimer['pgr_results'][k].pgr:.3f}",
        "405B CoT": f"{results_405b_cot['pgr_results'][k].pgr:.3f}",
        "70B Baseline": f"{results_70b['pgr_results'][k].pgr:.3f}",
        "70B Disclaimer": f"{results_70b_disclaimer['pgr_results'][k].pgr:.3f}",
        "70B CoT": f"{results_70b_cot['pgr_results'][k].pgr:.3f}",
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*80)
print("COMPLETE RESULTS SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))
```

---

## Summary of Key Findings

### Model Scale Matters

- **405B**: Near-perfect robustness throughout
- **70B**: Degrades at high K, sensitive to interventions

### Disclaimer Effect

- **405B**: Minimal effect (±0.02)
- **70B**: K-dependent reversal (helps at low K, hurts at high K)

### Chain-of-Thought Effect

- **405B**: Crossover pattern (hurts at K=0, helps at K>1)
- **70B**: Consistent degradation at K>0

### Critical Insight

**Large models (405B) can filter noisy reasoning, smaller models (70B) cannot.**

This is a novel finding about scaling and robustness to weak supervision with reasoning demonstrations!

---

## Estimated Total Runtime

- Dataset loading: < 1 min
- 405B Baseline: ~15-20 min
- 70B Baseline: ~10-15 min
- 405B Disclaimer: ~10 min
- 70B Disclaimer: ~8 min
- Weak CoT labels: ~5 min
- Gold CoT labels: ~10 min
- 405B CoT: ~10 min
- 70B CoT: ~8 min
- Visualization: < 1 min

**Total: ~75-90 minutes**

---

## Troubleshooting

### Issue: `asyncio.run()` error in notebook

**Solution:** Use `await` directly instead of `asyncio.run()` - Jupyter already has an event loop.

### Issue: Import errors

**Solution:** Ensure `sys.path.insert(0, str(project_root))` is run first.

### Issue: API rate limits

**Solution:** Reduce `max_parallel_requests` in config or add delays.

### Issue: Out of memory

**Solution:** Restart kernel and run experiments one at a time instead of all at once.
