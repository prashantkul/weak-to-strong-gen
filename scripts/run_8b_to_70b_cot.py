"""
Chain-of-Thought (CoT) Variant Experiment - 70B
Test if explicit reasoning helps 70B overcome weak label noise
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src import (
    Config, DatasetManager, ExperimentRunner, ResultsAnalyzer,
    ResultManager, ExperimentMetadata, get_model_pair
)


async def main():
    print("\n" + "="*70)
    print("8B→70B CHAIN-OF-THOUGHT EXPERIMENT")
    print("Does explicit reasoning help 70B overcome weak label noise?")
    print("="*70)

    # Configuration
    print("\n" + "-"*70)
    print("SETUP")
    print("-"*70)

    config = Config.from_env()
    config.setup_environment()

    # Use 8B→70B model pair
    pair = get_model_pair("8b_to_70b")
    config.weak_model = pair.weak_model
    config.strong_model = pair.strong_model

    print(f"Weak Model:  {config.weak_model}")
    print(f"Strong Model: {config.strong_model}")
    print(f"Parameter ratio: {pair.parameter_ratio}x")
    print(f"Temperature: {config.temperature} (deterministic)")
    print(f"Chain-of-Thought: ENABLED")

    # Load dataset
    print("\n" + "-"*70)
    print("LOADING DATASET")
    print("-"*70)

    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    print(f"✓ Test set: {len(test_data)} questions")
    print(f"✓ Few-shot pool: {len(few_shot_pool)} questions")

    # Test all K values to compare full PGR trajectories
    K_VALUES = [0, 2, 5, 10]
    print(f"\nTesting K={K_VALUES} to compare baseline vs cot curves")

    # Initialize experiment runner WITH COT
    runner = ExperimentRunner(config, use_cot=True)
    result_manager = ResultManager()

    # Load baseline results for comparison
    print("\n" + "="*70)
    print("LOADING BASELINE RESULTS & COT LABELS")
    print("="*70)

    import json
    baseline_exp = "results/8b_70b_baseline_20251102_132305/experiment.json"

    if Path(baseline_exp).exists():
        print(f"Loading baseline results from: {baseline_exp}")
        with open(baseline_exp) as f:
            baseline_data = json.load(f)

        # Load baseline PGR metrics
        baseline_pgr_metrics = {
            int(k): baseline_data['pgr_metrics'][k]
            for k in baseline_data['pgr_metrics'].keys()
        }
        print(f"✓ Loaded baseline PGR metrics for K={list(baseline_pgr_metrics.keys())}")
    else:
        raise FileNotFoundError(f"Baseline experiment not found: {baseline_exp}")

    # Load CoT weak labels (8B with reasoning)
    print("\nLoading CoT weak labels...")
    weak_cot_file = "data/cot_weak_labels/8b_cot_weak_labels_20251102_141149.json"
    if Path(weak_cot_file).exists():
        with open(weak_cot_file) as f:
            weak_cot_data = json.load(f)
        from src.model_evaluator import ModelResponse
        weak_cot_labels = [ModelResponse(**wl) for wl in weak_cot_data['weak_labels']]
        print(f"✓ Loaded {len(weak_cot_labels)} weak CoT labels (accuracy: {weak_cot_data['metadata']['accuracy']:.1%})")
    else:
        raise FileNotFoundError(f"Weak CoT labels not found: {weak_cot_file}")

    # Load CoT gold labels (405B with reasoning)
    print("\nLoading CoT gold labels...")
    gold_cot_file = "data/cot_gold_labels/405b_cot_gold_labels_20251102_141158.json"
    if Path(gold_cot_file).exists():
        with open(gold_cot_file) as f:
            gold_cot_data = json.load(f)
        gold_cot_labels = [ModelResponse(**gl) for gl in gold_cot_data['gold_labels']]
        print(f"✓ Loaded {len(gold_cot_labels)} gold CoT labels (accuracy: {gold_cot_data['metadata']['accuracy']:.1%})")
    else:
        raise FileNotFoundError(f"Gold CoT labels not found: {gold_cot_file}")

    # Run cot experiments for each K
    all_results = {}
    cot_pgr_results = {}

    for k in K_VALUES:
        print("\n" + "="*70)
        print(f"K = {k}")
        print("="*70)

        if k == 0:
            # K=0: No few-shot, should match baseline
            print("\nCoT at K=0 (no few-shot examples)")
            cot_result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.strong_model,
                experiment_name=f"strong_cot_k{k}"
            )
        else:
            # Few-shot with CoT - uses weak labels WITH reasoning
            print(f"\nStrong + Weak + COT (70B, K={k})")
            print(f"Using weak labels with reasoning from 8B model")
            cot_result = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=False,
                weak_responses=weak_cot_labels[:k],  # Use CoT labels with reasoning
                num_few_shot=k,
                experiment_name=f"strong_weak_cot_k{k}"
            )

        all_results[k] = {"strong_weak_cot": cot_result}

        # Calculate PGR for cot
        baseline_metrics = baseline_pgr_metrics[k]
        weak_baseline_acc = baseline_metrics['weak_baseline_accuracy']
        strong_gold_acc = baseline_metrics['strong_gold_accuracy']

        perf_gap = strong_gold_acc - weak_baseline_acc
        recovered_cot = cot_result.accuracy - weak_baseline_acc
        pgr_cot = recovered_cot / perf_gap if perf_gap != 0 else 1.0

        from src.results_analyzer import PGRMetrics
        pgr_metrics = PGRMetrics(
            weak_baseline_accuracy=weak_baseline_acc,
            strong_gold_accuracy=strong_gold_acc,
            strong_weak_accuracy=cot_result.accuracy,
            performance_gap=perf_gap,
            recovered_gap=recovered_cot,
            pgr=pgr_cot,
            pgr_percentage=f"{pgr_cot*100:.1f}%"
        )
        cot_pgr_results[k] = pgr_metrics

        # Show comparison
        baseline_pgr = baseline_metrics['pgr']
        print(f"\n{'─'*70}")
        print(f"K={k} Results:")
        print(f"  Baseline PGR:   {baseline_pgr:.3f}")
        print(f"  CoT PGR: {pgr_cot:.3f}")
        print(f"  Change:         {pgr_cot - baseline_pgr:+.3f}")
        print(f"{'─'*70}")

    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    import pandas as pd
    summary_data = []
    for k in K_VALUES:
        baseline_pgr = baseline_pgr_metrics[k]['pgr']
        cot_pgr = cot_pgr_results[k].pgr
        summary_data.append({
            "K": k,
            "Baseline PGR": f"{baseline_pgr:.3f}",
            "CoT PGR": f"{cot_pgr:.3f}",
            "Change": f"{cot_pgr - baseline_pgr:+.3f}",
            "Baseline Acc": f"{baseline_pgr_metrics[k]['strong_weak_accuracy']:.2%}",
            "CoT Acc": f"{cot_pgr_results[k].strong_weak_accuracy:.2%}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Check if cot helps
    changes = [cot_pgr_results[k].pgr - baseline_pgr_metrics[k]['pgr'] for k in K_VALUES]
    avg_change = sum(changes) / len(changes)
    max_change = max(changes)
    max_change_k = K_VALUES[changes.index(max_change)]

    if avg_change > 0.03:
        print(f"\n✓ SIGNIFICANT IMPROVEMENT: Chain-of-Thought helped 70B!")
        print(f"  Average PGR change: {avg_change:+.3f}")
        print(f"  Largest improvement at K={max_change_k}: {max_change:+.3f}")
        print(f"  Explicit reasoning helps mid-sized models overcome noise")
        print(f"\n  Key insight: 70B benefits from CoT more than 405B")
    elif avg_change < -0.02:
        print(f"\n✗ DEGRADATION: Chain-of-Thought hurt 70B performance")
        print(f"  Average PGR change: {avg_change:+.3f}")
        print(f"  Reasoning overhead may have reduced performance")
    else:
        print(f"\n→ MINIMAL EFFECT: Chain-of-Thought had small impact on 70B")
        print(f"  Average PGR change: {avg_change:+.3f}")
        print(f"  CoT doesn't significantly help or hurt 70B")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = result_manager.create_experiment_id(
        weak_model=config.weak_model,
        strong_model=config.strong_model,
        variant="cot",
        timestamp=timestamp
    )

    metadata = ExperimentMetadata(
        experiment_id=exp_id,
        timestamp=timestamp,
        weak_model=config.weak_model,
        strong_model=config.strong_model,
        dataset_name="truthfulqa_cot",
        test_size=len(test_data),
        k_values=K_VALUES,
        temperature=config.temperature,
        seed=42
    )

    result_manager.save_experiment(
        experiment_id=exp_id,
        metadata=metadata,
        all_results=all_results,
        pgr_results=cot_pgr_results,
        weak_labels=weak_cot_labels  # Save CoT weak labels
    )

    print(f"✓ Results saved to: results/{exp_id}/")

    # Visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    import matplotlib.pyplot as plt

    k_vals = K_VALUES
    baseline_pgrs = [baseline_pgr_metrics[k]['pgr'] for k in k_vals]
    cot_pgrs = [cot_pgr_results[k].pgr for k in k_vals]
    baseline_accs = [baseline_pgr_metrics[k]['strong_weak_accuracy'] for k in k_vals]
    cot_accs = [cot_pgr_results[k].strong_weak_accuracy for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: PGR curves
    ax1.plot(k_vals, baseline_pgrs, 'o-', label='Baseline (no cot)',
             linewidth=2, markersize=8, color='#d62728')
    ax1.plot(k_vals, cot_pgrs, 's-', label='With CoT',
             linewidth=2, markersize=8, color='#9467bd')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Perfect Recovery')
    ax1.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax1.set_ylabel('PGR', fontsize=12)
    ax1.set_title('70B: Baseline vs CoT PGR', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_vals)
    ax1.set_ylim(0.80, 1.05)

    # Plot 2: Accuracy curves
    ax2.plot(k_vals, baseline_accs, 'o-', label='Baseline (no cot)',
             linewidth=2, markersize=8, color='#d62728')
    ax2.plot(k_vals, cot_accs, 's-', label='With CoT',
             linewidth=2, markersize=8, color='#9467bd')
    ax2.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('70B: Strong+Weak Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_vals)

    plt.tight_layout()

    plot_path = Path(f"results/{exp_id}/cot_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    plt.show()

    print("\n" + "="*70)
    print("✓ 8B→70B CHAIN-OF-THOUGHT EXPERIMENT COMPLETE")
    print("="*70)

    print(f"\nComparison summary:")
    print(f"  Average PGR change: {avg_change:+.3f}")
    print(f"  Largest improvement at K={max_change_k}: {max_change:+.3f}")
    print(f"\nNext: Compare 405B vs 70B cot effectiveness")


if __name__ == "__main__":
    asyncio.run(main())
