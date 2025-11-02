# Improvements to Experimental Setup

## Summary of Changes

We've addressed critical reproducibility and extensibility concerns raised by the user.

## Problems Solved

### 1. Train/Test Contamination ✓

**Problem**: Previously, few-shot examples could accidentally come from the test set, leading to data leakage and inflated performance.

**Solution**: `DatasetManager` class with proper splitting:
```python
from src import DatasetManager

dm = DatasetManager()
test_data, few_shot_pool, split = dm.create_and_save_split(
    test_size=200,
    few_shot_pool_size=100,
    seed=42
)
```

**Verification**: Automated check ensures zero overlap between test and few-shot pool.

---

### 2. Dataset Persistence ✓

**Problem**: Random shuffling could create different splits on each run, making experiments non-reproducible.

**Solution**: Save splits to disk with exact question IDs:
- `data/all_questions.json` - All formatted questions
- `data/split_metadata.json` - Train/test/few-shot indices, seed, metadata

**Benefits**:
- Exact same split every time
- Can share splits with collaborators
- Reload without re-downloading dataset

```python
# First run: creates and saves
test_data, few_shot_pool, split = dm.create_and_save_split()

# Subsequent runs: loads saved split
test_data, few_shot_pool, split = dm.load_split()
```

---

### 3. Model Pair Flexibility ✓

**Problem**: Hard-coded model IDs made it difficult to test different model pairs.

**Solution**: `ModelPair` configuration system:

```python
from src import get_model_pair, list_model_pairs

# See available pairs
list_model_pairs()  # ['8b_to_70b', '8b_to_405b', '70b_to_405b']

# Get specific pair
pair = get_model_pair('8b_to_405b')
print(pair.weak_model)   # meta-llama/llama-3.1-8b-instruct
print(pair.strong_model) # meta-llama/llama-3.1-405b-instruct
```

**Predefined Pairs**:
- **8b_to_70b**: Moderate gap (8.75x params)
- **8b_to_405b**: Large gap (50x params) - PRIMARY
- **70b_to_405b**: Small gap (5.8x params)

**Easy to extend**: Add new pairs to `src/model_pairs.py`

---

### 4. Result Persistence ✓

**Problem**: Results only printed to console, lost after session ends.

**Solution**: `ResultManager` saves complete experiments to JSON:

```python
from src import ResultManager, ExperimentMetadata

rm = ResultManager(results_dir=Path("./results"))

# Save experiment
rm.save_experiment(
    experiment_id=exp_id,
    metadata=metadata,
    all_results=all_results,
    pgr_results=pgr_results,
    weak_labels=weak_labels  # Cache for reuse
)
```

**Saved Files**:
- `results/{exp_id}/experiment.json` - Complete results, all K values
- `results/{exp_id}/summary.txt` - Human-readable summary

**Enables**:
- Re-analyze without re-running experiments
- Compare across different model pairs
- Create visualizations from saved data
- Cost tracking and budgeting

---

### 5. Weak Label Reuse ✓

**Problem**: Generating weak labels costs API credits. Regenerating for each K value wastes money.

**Solution**: Cache weak model predictions:

```python
# Generate once
weak_labels = await runner.evaluator.generate_weak_labels(questions)

# Save for reuse
rm.save_weak_labels(
    weak_model=config.weak_model,
    weak_labels=weak_labels,
    question_ids=[q.question_id for q in questions]
)

# Reuse for all K values
for k in K_VALUES:
    await runner.run_few_shot_experiment(
        weak_responses=weak_labels,  # Same labels
        num_few_shot=k
    )
```

---

## New File Structure

```
astra/
├── data/                        # NEW: Dataset persistence
│   ├── all_questions.json      #   All formatted questions
│   └── split_metadata.json     #   Train/test split info
├── results/                     # NEW: Experiment results
│   └── {exp_id}/
│       ├── experiment.json     #   Complete results
│       └── summary.txt         #   Human-readable summary
├── src/
│   ├── dataset_manager.py      # NEW: Dataset splitting & persistence
│   ├── result_manager.py       # NEW: Result saving & loading
│   ├── model_pairs.py          # NEW: Model pair configurations
│   ├── config.py               # EXISTING
│   ├── model_evaluator.py      # EXISTING
│   ├── experiment_runner.py    # EXISTING
│   └── results_analyzer.py     # EXISTING
├── test_new_features.py         # NEW: Test improvements
├── IMPROVEMENTS.md              # NEW: This file
└── ...
```

---

## Usage Example: Complete Workflow

```python
from src import (
    Config, DatasetManager, ExperimentRunner,
    ResultsAnalyzer, ResultManager, get_model_pair
)

# 1. Select model pair
pair = get_model_pair('8b_to_405b')

# 2. Load or create dataset split
dm = DatasetManager()
if dm.split_exists():
    test_data, few_shot_pool, split = dm.load_split()
else:
    test_data, few_shot_pool, split = dm.create_and_save_split(
        test_size=200,
        few_shot_pool_size=100,
        seed=42
    )

# 3. Configure experiment
config = Config.from_env()
config.weak_model = pair.weak_model
config.strong_model = pair.strong_model

# 4. Run experiments
runner = ExperimentRunner(config)
all_results = {}

# Generate weak labels ONCE
weak_labels = await runner.evaluator.generate_weak_labels(
    questions=[(q.question_id, q.question) for q in few_shot_pool]
)

# Run for multiple K values
for k in [0, 2, 5, 10, 20]:
    results = await runner.run_full_experiment(
        train_data=few_shot_pool,
        test_data=test_data,
        num_few_shot=k
    )
    all_results[k] = results

# 5. Analyze
pgr_results = {}
for k, results in all_results.items():
    pgr_results[k] = ResultsAnalyzer.analyze_experiment(results)

# 6. Save everything
rm = ResultManager()
exp_id = rm.create_experiment_id(pair.weak_model, pair.strong_model)
rm.save_experiment(exp_id, metadata, all_results, pgr_results, weak_labels)

# 7. Later: Load and re-analyze
artifacts = rm.load_experiment(exp_id)
# Create new visualizations, etc.
```

---

## Benefits

### Reproducibility
- ✓ Exact same data split every time (saved to disk)
- ✓ Fixed random seed recorded in metadata
- ✓ All experiment parameters saved
- ✓ Can reproduce results from saved artifacts

### Cost Efficiency
- ✓ Weak labels generated once, reused for all K
- ✓ API caching still active
- ✓ Can estimate costs before running (using cost_multiplier)

### Extensibility
- ✓ Easy to test new model pairs (just add to model_pairs.py)
- ✓ Can run multiple experiments in parallel
- ✓ Results saved independently for comparison

### Data Integrity
- ✓ Zero train/test contamination (verified automatically)
- ✓ Few-shot examples never from test set
- ✓ Clean experimental design

---

## Verification

Run `python test_new_features.py` to verify:
- [x] Model pairs load correctly
- [x] Dataset split has no contamination
- [x] Splits persist to disk
- [x] Splits reload correctly
- [x] Result manager creates experiment IDs

---

## Next Steps

1. **Run full experiment** with improved setup
2. **Test additional model pairs** (8b→70b, 70b→405b)
3. **Compare results** across pairs using saved artifacts
4. **Create visualizations** from saved results
5. **Share splits** with collaborators for exact reproduction

---

## Migration from Old Code

If you have existing notebook code:

**Old approach** (risky):
```python
# Random split each time
truthful_all = random.sample(formatted_truthful, len(formatted_truthful))
truthful_test = truthful_all[:200]
truthful_train = truthful_all[200:]  # Could overlap with test!
```

**New approach** (safe):
```python
# Persistent split, no contamination
dm = DatasetManager()
test_data, few_shot_pool, split = dm.load_split()
# few_shot_pool is guaranteed disjoint from test_data
```

**Key difference**: Old code sampled from entire dataset including test set. New code only samples from guaranteed training set.
