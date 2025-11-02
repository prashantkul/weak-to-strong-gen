"""
Full weak-to-strong generalization experiment with K-sweep
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
    print("WEAK-TO-STRONG GENERALIZATION EXPERIMENT")
    print("In-Context Learning with Label Noise")
    print("="*70)

    # Configuration
    print("\n" + "-"*70)
    print("SETUP")
    print("-"*70)

    config = Config.from_env()
    config.setup_environment()

    # Use model pair
    pair = get_model_pair("8b_to_405b")
    config.weak_model = pair.weak_model
    config.strong_model = pair.strong_model

    print(f"Weak Model:  {config.weak_model}")
    print(f"Strong Model: {config.strong_model}")
    print(f"Temperature: {config.temperature} (deterministic)")
    print(f"Seed: 42")

    # Load dataset
    print("\n" + "-"*70)
    print("LOADING DATASET")
    print("-"*70)

    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    print(f"✓ Test set: {len(test_data)} questions")
    print(f"✓ Few-shot pool: {len(few_shot_pool)} questions")
    print(f"✓ No contamination verified")

    # K values to sweep
    K_VALUES = [0, 2, 5, 10, 20]
    print(f"\nK-sweep: {K_VALUES}")

    # Initialize experiment runner
    runner = ExperimentRunner(config)
    result_manager = ResultManager()

    # Generate weak labels ONCE for the few-shot pool
    print("\n" + "="*70)
    print("STEP 1: GENERATING WEAK LABELS")
    print("="*70)

    max_k = max(K_VALUES)
    pool_for_weak_labels = few_shot_pool[:max_k]
    print(f"\nGenerating weak labels for {len(pool_for_weak_labels)} questions...")

    weak_labels = await runner.evaluator.generate_weak_labels(
        questions=[(q.question_id, q.question) for q in pool_for_weak_labels],
        verbose=True
    )

    print(f"✓ Generated {len(weak_labels)} weak labels")

    # Calculate weak label accuracy
    weak_correct = sum(1 for wl, q in zip(weak_labels, pool_for_weak_labels) if wl.answer == q.answer)
    weak_acc = weak_correct / len(weak_labels)
    print(f"✓ Weak model accuracy on few-shot pool: {weak_acc:.2%} ({weak_correct}/{len(weak_labels)})")

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

            print("\nExperiment 2: Strong baseline (405B, K=0)")
            strong_result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.strong_model,
                experiment_name=f"strong_baseline_k{k}"
            )

            all_results[k] = {
                "weak_baseline": weak_result,
                "strong_gold": strong_result,
                "strong_weak": strong_result  # Same as gold at K=0
            }

        else:
            # Few-shot experiments
            print(f"\nExperiment 2: Strong + Gold (405B, K={k})")
            strong_gold = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=True,
                num_few_shot=k,
                experiment_name=f"strong_gold_k{k}"
            )

            print(f"\nExperiment 3: Strong + Weak (405B, K={k})")
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
                "weak_baseline": all_results[0]["weak_baseline"],  # Reuse K=0
                "strong_gold": strong_gold,
                "strong_weak": strong_weak
            }

        # Calculate PGR for this K
        pgr = ResultsAnalyzer.analyze_experiment(all_results[k])
        pgr_results[k] = pgr

        print(f"\n{'─'*70}")
        print(f"PGR for K={k}: {pgr.pgr:.3f} ({pgr.pgr_percentage})")
        print(f"{'─'*70}")

    # Final summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    import pandas as pd
    summary_data = []
    for k in K_VALUES:
        pgr = pgr_results[k]
        summary_data.append({
            "K": k,
            "Weak": f"{pgr.weak_baseline_accuracy:.2%}",
            "Strong+Gold": f"{pgr.strong_gold_accuracy:.2%}",
            "Strong+Weak": f"{pgr.strong_weak_accuracy:.2%}",
            "PGR": f"{pgr.pgr:.3f}",
            "PGR%": pgr.pgr_percentage
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = result_manager.create_experiment_id(
        weak_model=config.weak_model,
        strong_model=config.strong_model,
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
    ax1.plot(k_vals, strong_gold_accs, 's-', label='Strong + Gold (405B)', linewidth=2, markersize=8)
    ax1.plot(k_vals, strong_weak_accs, '^-', label='Strong + Weak (405B)', linewidth=2, markersize=8)
    ax1.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Performance vs Few-Shot Examples', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_vals)

    # Plot 2: PGR
    ax2.plot(k_vals, pgr_vals, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Recovery')
    ax2.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label='No Recovery')
    ax2.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax2.set_ylabel('PGR', fontsize=12)
    ax2.set_title('Performance Gap Recovered', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_vals)

    plt.tight_layout()

    plot_path = Path(f"results/{exp_id}/pgr_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    plt.show()

    print("\n" + "="*70)
    print("✓ EXPERIMENT COMPLETE")
    print("="*70)

    print(f"\nKey Findings:")
    print(f"  Weak baseline: {weak_accs[0]:.2%}")
    print(f"  Best strong+gold: {max(strong_gold_accs):.2%} at K={k_vals[strong_gold_accs.index(max(strong_gold_accs))]}")
    print(f"  Best PGR: {max(pgr_vals):.3f} at K={k_vals[pgr_vals.index(max(pgr_vals))]}")

    print(f"\nResults directory: results/{exp_id}/")
    print(f"  - experiment.json (complete results)")
    print(f"  - summary.txt (human-readable)")
    print(f"  - pgr_analysis.png (visualizations)")


if __name__ == "__main__":
    asyncio.run(main())
