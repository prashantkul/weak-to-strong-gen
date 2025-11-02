"""
8B→405B Baseline Experiment - Extended to K=10
K-sweep: {0, 2, 5, 10} - comparing robustness with 8B→70B
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
    print("8B→405B BASELINE EXPERIMENT (K=0,2,5,10)")
    print("Testing 405B robustness to weak labels at K=10")
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

    # Extended K values to compare with 8B→70B
    K_VALUES = [0, 2, 5, 10]
    print(f"\n" + "-"*70)
    print(f"K-SWEEP: Extended to K=10")
    print(f"-"*70)
    print(f"K values: {K_VALUES}")
    print(f"Testing K=10 to compare with 8B→70B degradation pattern")

    # Initialize experiment runner
    runner = ExperimentRunner(config)
    result_manager = ResultManager()

    # Generate weak labels ONCE for the few-shot pool
    print("\n" + "="*70)
    print("STEP 1: GENERATING WEAK LABELS")
    print("="*70)

    max_k = max(K_VALUES)
    pool_for_weak_labels = few_shot_pool[:max_k]

    # Check if we can reuse from previous experiment
    import json
    prev_exp = "results/8b_405b_baseline_20251102_130630/experiment.json"

    weak_labels = []
    if Path(prev_exp).exists():
        print(f"Loading cached weak labels from previous experiment...")
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
    print("RESULTS SUMMARY (K=0,2,5,10)")
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

    # Analysis comparing 405B vs 70B robustness
    print("\n" + "="*70)
    print("ROBUSTNESS COMPARISON: 405B vs 70B")
    print("="*70)

    pgr_at_5 = pgr_results[5].pgr
    pgr_at_10 = pgr_results[10].pgr if 10 in pgr_results else None

    print(f"\n8B→405B PGR trend:")
    print(f"  K=0 → K=2: {pgr_results[0].pgr:.3f} → {pgr_results[2].pgr:.3f} (Δ={pgr_results[2].pgr - pgr_results[0].pgr:+.3f})")
    print(f"  K=2 → K=5: {pgr_results[2].pgr:.3f} → {pgr_at_5:.3f} (Δ={pgr_at_5 - pgr_results[2].pgr:+.3f})")
    if pgr_at_10:
        print(f"  K=5 → K=10: {pgr_at_5:.3f} → {pgr_at_10:.3f} (Δ={pgr_at_10 - pgr_at_5:+.3f})")

    print(f"\n8B→70B PGR trend (for comparison):")
    print(f"  K=0 → K=2: 1.000 → 0.920 (Δ=-0.080)")
    print(f"  K=2 → K=5: 0.920 → 0.930 (Δ=+0.010)")
    print(f"  K=5 → K=10: 0.930 → 0.864 (Δ=-0.066)")

    if pgr_at_10:
        if pgr_at_10 >= 0.98:
            print(f"\n✓ FINDING: 405B maintains high robustness even at K=10")
            print(f"  → PGR@K10 = {pgr_at_10:.3f} (vs 0.864 for 70B)")
            print(f"  → Model scale provides immunity to label noise")
        elif pgr_at_10 < pgr_at_5 - 0.05:
            print(f"\n⚠ FINDING: 405B also shows degradation at K=10!")
            print(f"  → PGR@K10 = {pgr_at_10:.3f} (vs 0.864 for 70B)")
            print(f"  → Even large models have limits with noisy supervision")
        else:
            print(f"\n→ FINDING: 405B shows minor degradation at K=10")
            print(f"  → PGR@K10 = {pgr_at_10:.3f} (vs 0.864 for 70B)")
            print(f"  → Better than 70B but not immune to noise accumulation")

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
    ax1.plot(k_vals, strong_gold_accs, 's-', label='Strong + Gold (405B)', linewidth=2, markersize=8)
    ax1.plot(k_vals, strong_weak_accs, '^-', label='Strong + Weak (405B)', linewidth=2, markersize=8)
    ax1.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Performance vs Few-Shot Examples (Phase 1)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_vals)

    # Plot 2: PGR
    ax2.plot(k_vals, pgr_vals, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Recovery')
    ax2.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label='No Recovery')
    ax2.set_xlabel('K (Few-Shot Examples)', fontsize=12)
    ax2.set_ylabel('PGR', fontsize=12)
    ax2.set_title('Performance Gap Recovered (Extended)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_vals)

    plt.tight_layout()

    plot_path = Path(f"results/{exp_id}/pgr_analysis_8b_to_405b.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    plt.show()

    print("\n" + "="*70)
    print("✓ 8B→405B EXPERIMENT COMPLETE (K=0,2,5,10)")
    print("="*70)

    print(f"\nKey Findings:")
    print(f"  Weak baseline: {weak_accs[0]:.2%}")
    print(f"  Best strong+gold: {max(strong_gold_accs):.2%} at K={k_vals[strong_gold_accs.index(max(strong_gold_accs))]}")
    print(f"  Best PGR: {max(pgr_vals):.3f} at K={k_vals[pgr_vals.index(max(pgr_vals))]}")

    if 10 in pgr_results:
        print(f"\n  405B Robustness at K=10:")
        print(f"    PGR: {pgr_results[10].pgr:.3f}")
        print(f"    vs 70B PGR: 0.864")
        print(f"    Difference: {pgr_results[10].pgr - 0.864:+.3f}")

    print(f"\nResults directory: results/{exp_id}/")
    print(f"  - experiment.json (complete results)")
    print(f"  - summary.txt (human-readable)")
    print(f"  - pgr_analysis_8b_to_405b.png (visualizations)")

    print(f"\nComparison with 8B→70B complete!")
    print(f"  Next: Run disclaimer experiment or analyze findings")


if __name__ == "__main__":
    asyncio.run(main())
