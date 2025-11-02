"""
8B→70B Model Pair Experiment
Test PGR with smaller performance gap (8.75x parameters vs 50x)
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
    print("8B→70B MODEL PAIR EXPERIMENT")
    print("Testing PGR with smaller performance gap")
    print("="*70)

    # Configuration
    print("\n" + "-"*70)
    print("SETUP")
    print("-"*70)

    config = Config.from_env()
    config.setup_environment()

    # Use 8B→70B pair (smaller gap)
    pair = get_model_pair("8b_to_70b")
    config.weak_model = pair.weak_model
    config.strong_model = pair.strong_model

    print(f"Weak Model:  {config.weak_model}")
    print(f"Strong Model: {config.strong_model}")
    print(f"Parameter ratio: {pair.parameter_ratio}x (vs 50x for 8B→405B)")
    print(f"Temperature: {config.temperature} (deterministic)")

    # Load dataset
    print("\n" + "-"*70)
    print("LOADING DATASET")
    print("-"*70)

    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    print(f"✓ Test set: {len(test_data)} questions")
    print(f"✓ Few-shot pool: {len(few_shot_pool)} questions")

    # Test K={0, 2, 5, 10} to see if PGR continues improving
    K_VALUES = [0, 2, 5, 10]
    print(f"\nK-sweep: {K_VALUES}")

    # Initialize experiment runner
    runner = ExperimentRunner(config)
    result_manager = ResultManager()

    # Generate weak labels (8B model, same as before)
    print("\n" + "="*70)
    print("GENERATING WEAK LABELS (8B)")
    print("="*70)

    max_k = max(K_VALUES)
    pool_for_weak_labels = few_shot_pool[:max_k]

    # Check if we can reuse from previous experiment
    import json
    prev_exp = "results/8b_405b_baseline_20251102_130630/experiment.json"

    weak_labels = []
    if Path(prev_exp).exists():
        print(f"Loading cached weak labels from previous 8B experiment...")
        with open(prev_exp) as f:
            prev_data = json.load(f)

        from src.model_evaluator import ModelResponse
        weak_labels_data = prev_data.get('weak_labels_cache', [])
        weak_labels = [ModelResponse(**wl) for wl in weak_labels_data]
        print(f"✓ Loaded {len(weak_labels)} cached weak labels")

        # Generate additional labels if needed
        if len(weak_labels) < max_k:
            num_needed = max_k - len(weak_labels)
            print(f"\nGenerating {num_needed} additional weak labels (K={len(weak_labels)}→{max_k})...")
            additional_questions = pool_for_weak_labels[len(weak_labels):max_k]
            additional_labels = await runner.evaluator.generate_weak_labels(
                questions=[(q.question_id, q.question) for q in additional_questions],
                verbose=True
            )
            weak_labels.extend(additional_labels)
            print(f"✓ Total weak labels: {len(weak_labels)}")
    else:
        print("Generating fresh weak labels...")
        weak_labels = await runner.evaluator.generate_weak_labels(
            questions=[(q.question_id, q.question) for q in pool_for_weak_labels],
            verbose=True
        )

    # Run experiments for each K
    all_results = {}
    pgr_results = {}

    for k in K_VALUES:
        print("\n" + "="*70)
        print(f"K = {k}")
        print("="*70)

        if k == 0:
            # Baseline: No few-shot
            print("\nExperiment 1: Weak baseline (8B, K=0)")
            weak_result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.weak_model,
                experiment_name=f"weak_baseline_k{k}"
            )

            print("\nExperiment 2: Strong baseline (70B, K=0)")
            strong_result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.strong_model,
                experiment_name=f"strong_baseline_k{k}"
            )

            all_results[k] = {
                "weak_baseline": weak_result,
                "strong_gold": strong_result,
                "strong_weak": strong_result
            }

        else:
            # Few-shot experiments
            print(f"\nExperiment 2: Strong + Gold (70B, K={k})")
            strong_gold = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=True,
                num_few_shot=k,
                experiment_name=f"strong_gold_k{k}"
            )

            print(f"\nExperiment 3: Strong + Weak (70B, K={k})")
            strong_weak = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=False,
                weak_responses=weak_labels,
                num_few_shot=k,
                experiment_name=f"strong_weak_k{k}"
            )

            all_results[k] = {
                "weak_baseline": all_results[0]["weak_baseline"],
                "strong_gold": strong_gold,
                "strong_weak": strong_weak
            }

        # Calculate PGR
        pgr = ResultsAnalyzer.analyze_experiment(all_results[k])
        pgr_results[k] = pgr

        print(f"\n{'─'*70}")
        print(f"PGR for K={k}: {pgr.pgr:.3f} ({pgr.pgr_percentage})")
        print(f"{'─'*70}")

    # Summary
    print("\n" + "="*70)
    print("8B→70B RESULTS SUMMARY")
    print("="*70)

    import pandas as pd
    summary_data = []
    for k in K_VALUES:
        pgr = pgr_results[k]
        summary_data.append({
            "K": k,
            "Weak (8B)": f"{pgr.weak_baseline_accuracy:.2%}",
            "Strong+Gold (70B)": f"{pgr.strong_gold_accuracy:.2%}",
            "Strong+Weak (70B)": f"{pgr.strong_weak_accuracy:.2%}",
            "PGR": f"{pgr.pgr:.3f}",
            "PGR%": pgr.pgr_percentage
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Comparison with 8B→405B
    print("\n" + "="*70)
    print("COMPARISON: 8B→70B vs 8B→405B")
    print("="*70)

    print(f"\n{'Metric':<30} {'8B→70B':<15} {'8B→405B':<15}")
    print("-"*70)
    print(f"{'Parameter ratio':<30} {'8.75x':<15} {'50x':<15}")
    print(f"{'Weak baseline (8B)':<30} {pgr_results[0].weak_baseline_accuracy:<14.2%} {'58.5%':<15}")
    print(f"{'Strong zero-shot':<30} {pgr_results[0].strong_gold_accuracy:<14.2%} {'86.0%':<15}")
    print(f"{'Strong + Gold (K=5)':<30} {pgr_results[5].strong_gold_accuracy:<14.2%} {'90.0%':<15}")
    print(f"{'Strong + Weak (K=5)':<30} {pgr_results[5].strong_weak_accuracy:<14.2%} {'90.0%':<15}")
    print(f"{'PGR (K=5)':<30} {pgr_results[5].pgr:<14.3f} {'1.000':<15}")

    if 10 in pgr_results:
        print(f"{'Strong + Gold (K=10)':<30} {pgr_results[10].strong_gold_accuracy:<14.2%} {'N/A':<15}")
        print(f"{'Strong + Weak (K=10)':<30} {pgr_results[10].strong_weak_accuracy:<14.2%} {'N/A':<15}")
        print(f"{'PGR (K=10)':<30} {pgr_results[10].pgr:<14.3f} {'N/A':<15}")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = result_manager.create_experiment_id(
        weak_model=config.weak_model,
        strong_model=config.strong_model,
        variant="baseline",
        timestamp=timestamp
    )

    metadata = ExperimentMetadata(
        experiment_id=exp_id,
        timestamp=timestamp,
        weak_model=config.weak_model,
        strong_model=config.strong_model,
        dataset_name="truthfulqa",
        test_size=len(test_data),
        k_values=K_VALUES,
        temperature=config.temperature,
        seed=42
    )

    result_manager.save_experiment(
        experiment_id=exp_id,
        metadata=metadata,
        all_results=all_results,
        pgr_results=pgr_results,
        weak_labels=weak_labels
    )

    print(f"✓ Results saved to: results/{exp_id}/")

    # Visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    import matplotlib.pyplot as plt

    k_vals = K_VALUES
    weak_accs = [pgr_results[k].weak_baseline_accuracy for k in k_vals]
    strong_gold_accs = [pgr_results[k].strong_gold_accuracy for k in k_vals]
    strong_weak_accs = [pgr_results[k].strong_weak_accuracy for k in k_vals]
    pgr_vals = [pgr_results[k].pgr for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy
    ax1.plot(k_vals, weak_accs, 'o-', label='Weak (8B)', linewidth=2, markersize=8)
    ax1.plot(k_vals, strong_gold_accs, 's-', label='Strong + Gold (70B)', linewidth=2, markersize=8)
    ax1.plot(k_vals, strong_weak_accs, '^-', label='Strong + Weak (70B)', linewidth=2, markersize=8)
    ax1.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('8B→70B: Performance vs Few-Shot', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_vals)

    # Plot 2: PGR
    ax2.plot(k_vals, pgr_vals, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Recovery')
    ax2.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label='No Recovery')
    ax2.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax2.set_ylabel('PGR', fontsize=12)
    ax2.set_title('8B→70B: Performance Gap Recovered', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_vals)

    plt.tight_layout()

    plot_path = Path(f"results/{exp_id}/pgr_analysis_8b_to_70b.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    plt.show()

    print("\n" + "="*70)
    print("✓ 8B→70B EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
