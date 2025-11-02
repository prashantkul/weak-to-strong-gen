# Weak-to-Strong Generalization with In-Context Learning

This project investigates how strong language models perform when learning from weak supervision via in-context learning, and how metacognitive interventions affect this robustness.

## Overview

**Research Question**: How do large language models perform when learning from few-shot examples labeled by weaker models, and how do metacognitive interventions (disclaimer, chain-of-thought) affect this robustness?

**Model Pairs Tested**:
- 8B → 405B (50× parameter scaling)
- 8B → 70B (8.75× parameter scaling)

**Interventions Tested**:
1. **Baseline**: Standard few-shot learning with weak labels
2. **Disclaimer**: Metacognitive prompt warning about label quality
3. **Chain-of-Thought (CoT)**: Few-shot examples include explicit reasoning

**Key Metric**: Performance Gap Recovered (PGR)
```
PGR = (Strong_weak - Weak_baseline) / (Strong_gold - Weak_baseline)
```

## Key Findings

🔑 **Model scale determines robustness**: 405B maintains near-perfect robustness (PGR ≥ 0.984) across all K values, while 70B degrades to 0.864 at K=10.

🔑 **K-dependent reversal effect**: Disclaimer helps at low K (K=2: +0.080 for 70B) but hurts at high K (K=10: -0.017 for 70B) - first documented reversal in metacognitive prompting.

🔑 **Crossover effect in CoT**: For 405B, CoT hurts at K=0 (-0.164) but helps at K>0 (+0.095 at K=2). For 70B, CoT consistently hurts at K>0.

🔑 **Reasoning filter threshold**: Large models (405B) can learn from imperfect reasoning, medium models (70B) cannot - reveals critical scaling threshold.

See [KEY_FINDINGS.md](KEY_FINDINGS.md) for comprehensive analysis.

## Project Structure

```
astra/
├── src/                                 # Core library
│   ├── __init__.py
│   ├── config.py                        # Configuration management
│   ├── dataset_manager.py               # TruthfulQA loading and splitting
│   ├── model_evaluator.py               # Model inference with CoT support
│   ├── experiment_runner.py             # Experiment orchestration
│   ├── results_analyzer.py              # PGR calculation and analysis
│   └── result_manager.py                # Save/load experiment results
│
├── scripts/                             # Experiment scripts
│   ├── run_8b_to_405b_baseline.py       # 405B baseline (K={0,2,5,10})
│   ├── run_8b_to_70b_baseline.py        # 70B baseline (K={0,2,5,10})
│   ├── run_8b_to_405b_disclaimer.py     # 405B disclaimer (K={0,2,5,10})
│   ├── run_8b_to_70b_disclaimer.py      # 70B disclaimer (K={0,2,5,10})
│   ├── run_8b_to_405b_cot.py            # 405B CoT (K={0,2,5,10})
│   ├── run_8b_to_70b_cot.py             # 70B CoT (K={0,2,5,10})
│   ├── generate_cot_weak_labels.py      # Generate 8B reasoning labels
│   └── generate_cot_gold_labels.py      # Generate 405B reasoning labels
│
├── notebook_experiments.py              # Jupyter-compatible async functions
├── test_notebook_functions.py           # Quick test of notebook setup
│
├── create_final_comparison.py           # Generate 6-panel visualization
│
├── Complete_Weak_to_Strong_Experiments.ipynb  # All experiments in one notebook
├── NOTEBOOK_GUIDE.md                    # Step-by-step Jupyter guide
├── KEY_FINDINGS.md                      # Comprehensive research findings
│
├── data/
│   ├── cot_weak_labels/                 # 8B reasoning demonstrations
│   └── cot_gold_labels/                 # 405B reasoning demonstrations
│
├── results/                             # Experiment outputs
│   ├── 8b_405b_baseline_*/              # 405B baseline results
│   ├── 8b_70b_baseline_*/               # 70B baseline results
│   ├── 8b_405b_disclaimer_*/            # 405B disclaimer results
│   ├── 8b_70b_disclaimer_*/             # 70B disclaimer results
│   ├── 8b_405b_cot_*/                   # 405B CoT results
│   ├── 8b_70b_cot_*/                    # 70B CoT results
│   └── final_comprehensive_comparison.png
│
├── cache/                               # Cached API responses
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup Instructions

### Local Development (Conda)

1. **Create conda environment:**
   ```bash
   conda create -n astra python=3.10
   conda activate astra
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API keys
   ```

4. **Run the notebook:**
   ```bash
   jupyter notebook Prashant_Kulkarni_final_icl_w2s_empty.ipynb
   ```

### Google Colab

1. **Upload the notebook** to Colab

2. **Set API keys** in one of two ways:
   - **Option A**: Set in cell-6 directly (already in notebook)
   - **Option B**: Use Colab Secrets (recommended):
     - Click on the key icon in the left sidebar
     - Add secrets:
       - `OPENROUTER_API_KEY`
       - `OPENROUTER_API_KEY_BACKUP`

3. **Upload src folder** or copy class definitions inline

4. **Run all cells**

## Experimental Design

### Key Parameters
- **Model Pairs**:
  - 8B → 405B (50× scaling)
  - 8B → 70B (8.75× scaling)
- **Temperature**: 0.0 (deterministic decoding)
- **Few-shot K values**: {0, 2, 5, 10}
- **Dataset**: TruthfulQA (200 test, 100 few-shot pool, no overlap)
- **Interventions**: Baseline, Disclaimer, Chain-of-Thought

### Experiments Run (6 total)
For each model pair (405B, 70B) × each intervention (baseline, disclaimer, CoT):
1. **Weak Baseline** (K=0): 8B model, no few-shot examples
2. **Strong + Gold** (K=0): Strong model, no few-shot examples
3. **For K∈{2,5,10}**:
   - Strong + Gold (K examples): Strong model with K gold-labeled examples
   - Strong + Weak (K examples): Strong model with K weak-labeled examples

### Intervention Details

**Baseline**: Standard few-shot learning
- Format: Question → Answer (letter only)
- No special prompting

**Disclaimer**: Metacognitive warning prompt
- Added to system prompt: "Note: The provided examples may contain errors. Please think critically and rely on your own knowledge."
- Tests if warning about label quality helps robustness

**Chain-of-Thought (CoT)**: Reasoning demonstrations
- Format: Question → Reasoning → Answer
- Weak labels: 8B model with CoT prompting (70% accuracy)
- Gold labels: 405B model with CoT prompting (75% accuracy)
- Few-shot examples include full reasoning chains

### PGR Interpretation
- **PGR = 1.0 (100%)**: Perfect robustness, strong model fully ignores weak labels
- **PGR > 1.0 (>100%)**: Superelicitation, weak labels help beyond gold labels
- **PGR ≥ 0.9 (≥90%)**: Excellent recovery
- **PGR ≥ 0.5 (≥50%)**: Good recovery
- **PGR ≥ 0.0 (≥0%)**: Partial recovery
- **PGR < 0.0 (<0%)**: Negative recovery, strong model worse than weak baseline

## Quick Start

### Option 1: Run from Jupyter Notebook (Recommended)

Open `Complete_Weak_to_Strong_Experiments.ipynb` and run all cells. This notebook includes:
- All 6 experiments (baseline, disclaimer, CoT × 405B, 70B)
- Dataset loading and verification
- Comprehensive visualization
- Key findings summary

See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) for detailed step-by-step instructions.

### Option 2: Run Individual Scripts

```bash
# 1. Setup environment
conda activate astra
export OPENROUTER_API_KEY="your_key"

# 2. Run baseline experiments
python scripts/run_8b_to_405b_baseline.py
python scripts/run_8b_to_70b_baseline.py

# 3. Run disclaimer experiments
python scripts/run_8b_to_405b_disclaimer.py
python scripts/run_8b_to_70b_disclaimer.py

# 4. Generate CoT labels
python scripts/generate_cot_weak_labels.py
python scripts/generate_cot_gold_labels.py

# 5. Run CoT experiments
python scripts/run_8b_to_405b_cot.py
python scripts/run_8b_to_70b_cot.py

# 6. Create comparison visualization
python create_final_comparison.py
```

### Option 3: Use as Python Library

```python
from src import Config, DatasetManager, ExperimentRunner

# Load configuration
config = Config.from_env()
config.setup_environment()

# Load dataset
dm = DatasetManager()
test_data, few_shot_pool, split = dm.load_split()

# Run experiment
runner = ExperimentRunner(config)
results = await runner.run_few_shot_experiment(
    train_data=few_shot_pool[:10],
    test_data=test_data,
    model_id=config.strong_model,
    use_gold_labels=False,
    weak_responses=weak_labels[:10],
    num_few_shot=10,
    experiment_name="strong_weak_k10"
)

print(f"Accuracy: {results.accuracy:.1%}")
```

## Key Features

✅ **Complete experimental suite**: 6 experiments (2 baselines, 2 disclaimers, 2 CoT)
✅ **Modular design**: Separate classes for evaluation, experiments, and analysis
✅ **Jupyter-compatible**: Async functions work directly in notebooks
✅ **Works locally & Colab**: Automatic environment detection
✅ **K-sweep support**: Test multiple few-shot sizes in one run
✅ **Deterministic**: Temperature=0 for reproducibility
✅ **Cached responses**: Avoid duplicate API calls
✅ **Comprehensive visualization**: 6-panel comparison of all interventions
✅ **Detailed documentation**: KEY_FINDINGS.md with full analysis

## API Budget Management

- **Primary Key**: $300 budget
- **Backup Key**: $300 budget (use if primary exhausted)
- **Max Threads**: 50 concurrent requests
- **Caching**: Enabled to avoid duplicate charges

Monitor usage at: https://openrouter.ai/settings/keys

## Results Summary

All experimental results are saved in `results/` with timestamped directories. Each experiment includes:
- Full experiment metadata (config, timestamps, model IDs)
- Raw model responses (all predictions)
- Accuracy metrics (correct/total, percentage)
- PGR calculations (for K>0 experiments)
- Visualizations (accuracy curves, PGR curves)

### Best Results Achieved

| Model | Intervention | K | PGR | Notes |
|-------|--------------|---|-----|-------|
| 405B | CoT | 10 | **1.048** | Superelicitation! |
| 405B | Disclaimer | 2 | 1.000 | Perfect recovery |
| 405B | Baseline | 5 | 1.000 | Perfect recovery |
| 70B | Disclaimer | 2 | 1.000 | Perfect recovery |
| 70B | Baseline | 0 | 1.000 | No supervision |

See `results/final_comprehensive_comparison.png` for visual comparison of all 6 conditions.

## Reproducibility

All experiments use:
- **Fixed random seed**: 42 for all sampling
- **Deterministic decoding**: Temperature=0
- **Consistent dataset split**: Same 200 test questions, 100 few-shot pool
- **Prefix-based sampling**: K=2 is subset of K=5, which is subset of K=10
- **Cached responses**: Same prompt → cached result (no API re-calls)
- **Fixed model versions**: Llama 3.1 8B/70B/405B via OpenRouter

## Citation

If you use this code or findings in your research, please cite:

```
Weak-to-Strong Generalization with In-Context Learning:
Scale-Dependent Effects of Metacognitive Interventions
Research Project, November 2025
```

## Contributing

This is a research project. For questions or discussions about the findings, please open an issue.

## License

MIT License - See LICENSE file for details

## Contact

For questions about the experimental setup or results, please refer to [KEY_FINDINGS.md](KEY_FINDINGS.md) or open an issue in the repository.
