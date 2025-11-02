"""
Notebook-compatible experiment functions
These can be run directly in Jupyter without asyncio.run()
"""

from pathlib import Path
from datetime import datetime
from typing import List
import json

from src import (
    Config, DatasetManager, ExperimentRunner, ResultsAnalyzer,
    ResultManager, ExperimentMetadata, get_model_pair
)
from src.model_evaluator import ModelResponse
from src.results_analyzer import PGRMetrics


async def run_baseline_sweep(
    model_pair: str = "8b_to_405b",
    k_values: List[int] = None,
    save_results: bool = True
):
    """
    Run baseline experiments for a model pair across multiple K values.

    Usage in Jupyter:
        await run_baseline_sweep("8b_to_405b", k_values=[0, 2, 5, 10])

    Args:
        model_pair: One of "8b_to_405b", "8b_to_70b"
        k_values: List of K values to test
        save_results: Whether to save results to disk

    Returns:
        Dictionary with all results and PGR metrics
    """
    if k_values is None:
        k_values = [0, 2, 5, 10]

    print("\n" + "="*70)
    print(f"BASELINE EXPERIMENT: {model_pair.upper()}")
    print("="*70)

    # Setup
    config = Config.from_env()
    config.setup_environment()

    pair = get_model_pair(model_pair)
    config.weak_model = pair.weak_model
    config.strong_model = pair.strong_model

    print(f"\nWeak Model:  {config.weak_model}")
    print(f"Strong Model: {config.strong_model}")
    print(f"K values: {k_values}")

    # Load dataset
    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    print(f"\n✓ Test set: {len(test_data)} questions")
    print(f"✓ Few-shot pool: {len(few_shot_pool)} questions")

    # Initialize runner
    runner = ExperimentRunner(config)
    result_manager = ResultManager()

    # Generate weak labels
    print("\n" + "="*70)
    print("GENERATING WEAK LABELS")
    print("="*70)

    max_k = max(k_values)
    train_questions = [(q.question_id, q.question) for q in few_shot_pool[:max_k]]
    weak_labels = await runner.evaluator.generate_weak_labels(
        questions=train_questions,
        verbose=True
    )

    print(f"✓ Generated {len(weak_labels)} weak labels")

    # Run experiments for each K
    all_results = {}
    pgr_results = {}

    for k in k_values:
        print("\n" + "="*70)
        print(f"K = {k}")
        print("="*70)

        if k == 0:
            # K=0 baselines
            weak_result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.weak_model,
                experiment_name=f"weak_k{k}"
            )
            strong_result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.strong_model,
                experiment_name=f"strong_k{k}"
            )
            all_results[k] = {
                "weak_baseline": weak_result,
                "strong_gold": strong_result,
                "strong_weak": strong_result
            }
        else:
            # Few-shot experiments
            strong_gold = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=True,
                num_few_shot=k,
                experiment_name=f"strong_gold_k{k}"
            )

            strong_weak = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=False,
                weak_responses=weak_labels[:k],
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

        print(f"\nK={k} Results:")
        print(pgr)

    # Save if requested
    if save_results:
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
            k_values=k_values,
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

        print(f"\n✓ Results saved to: results/{exp_id}/")

    return {
        "all_results": all_results,
        "pgr_results": pgr_results,
        "weak_labels": weak_labels,
        "metadata": {
            "weak_model": config.weak_model,
            "strong_model": config.strong_model,
            "k_values": k_values
        }
    }


async def run_disclaimer_sweep(
    model_pair: str = "8b_to_405b",
    k_values: List[int] = None,
    baseline_exp_path: str = None,
    save_results: bool = True
):
    """
    Run disclaimer experiments.

    Usage in Jupyter:
        await run_disclaimer_sweep(
            "8b_to_70b",
            k_values=[0, 2, 5, 10],
            baseline_exp_path="results/8b_70b_baseline_20251102_132305/experiment.json"
        )

    Args:
        model_pair: One of "8b_to_405b", "8b_to_70b"
        k_values: List of K values to test
        baseline_exp_path: Path to baseline experiment.json for comparison
        save_results: Whether to save results

    Returns:
        Dictionary with results and comparison to baseline
    """
    if k_values is None:
        k_values = [0, 2, 5, 10]

    print("\n" + "="*70)
    print(f"DISCLAIMER EXPERIMENT: {model_pair.upper()}")
    print("="*70)

    # Setup
    config = Config.from_env()
    config.setup_environment()

    pair = get_model_pair(model_pair)
    config.weak_model = pair.weak_model
    config.strong_model = pair.strong_model

    print(f"\nWeak Model:  {config.weak_model}")
    print(f"Strong Model: {config.strong_model}")
    print(f"Disclaimer: ENABLED")

    # Load dataset
    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    # Load baseline results
    if baseline_exp_path and Path(baseline_exp_path).exists():
        print(f"\nLoading baseline from: {baseline_exp_path}")
        with open(baseline_exp_path) as f:
            baseline_data = json.load(f)
        baseline_pgr_metrics = {
            int(k): baseline_data['pgr_metrics'][k]
            for k in baseline_data['pgr_metrics'].keys()
        }
        weak_labels_data = baseline_data.get('weak_labels_cache', [])
        weak_labels = [ModelResponse(**wl) for wl in weak_labels_data]
        print(f"✓ Loaded baseline PGR metrics")
        print(f"✓ Loaded {len(weak_labels)} weak labels")
    else:
        raise ValueError(f"Baseline experiment not found: {baseline_exp_path}")

    # Initialize runner WITH disclaimer
    runner = ExperimentRunner(config, use_disclaimer=True)
    result_manager = ResultManager()

    # Run experiments
    all_results = {}
    disclaimer_pgr_results = {}

    for k in k_values:
        print("\n" + "="*70)
        print(f"K = {k}")
        print("="*70)

        if k == 0:
            result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.strong_model,
                experiment_name=f"strong_disclaimer_k{k}"
            )
        else:
            result = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=False,
                weak_responses=weak_labels[:k],
                num_few_shot=k,
                experiment_name=f"strong_weak_disclaimer_k{k}"
            )

        all_results[k] = {"strong_weak_disclaimer": result}

        # Calculate PGR
        baseline_metrics = baseline_pgr_metrics[k]
        weak_baseline_acc = baseline_metrics['weak_baseline_accuracy']
        strong_gold_acc = baseline_metrics['strong_gold_accuracy']
        perf_gap = strong_gold_acc - weak_baseline_acc
        recovered = result.accuracy - weak_baseline_acc
        pgr = recovered / perf_gap if perf_gap != 0 else 1.0

        pgr_metrics = PGRMetrics(
            weak_baseline_accuracy=weak_baseline_acc,
            strong_gold_accuracy=strong_gold_acc,
            strong_weak_accuracy=result.accuracy,
            performance_gap=perf_gap,
            recovered_gap=recovered,
            pgr=pgr,
            pgr_percentage=f"{pgr*100:.1f}%"
        )
        disclaimer_pgr_results[k] = pgr_metrics

        # Show comparison
        baseline_pgr = baseline_metrics['pgr']
        print(f"\nK={k} Comparison:")
        print(f"  Baseline PGR:   {baseline_pgr:.3f}")
        print(f"  Disclaimer PGR: {pgr:.3f}")
        print(f"  Change:         {pgr - baseline_pgr:+.3f}")

    # Save if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = result_manager.create_experiment_id(
            weak_model=config.weak_model,
            strong_model=config.strong_model,
            variant="disclaimer",
            timestamp=timestamp
        )

        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            timestamp=timestamp,
            weak_model=config.weak_model,
            strong_model=config.strong_model,
            dataset_name="truthfulqa_disclaimer",
            test_size=len(test_data),
            k_values=k_values,
            temperature=config.temperature,
            seed=42
        )

        result_manager.save_experiment(
            experiment_id=exp_id,
            metadata=metadata,
            all_results=all_results,
            pgr_results=disclaimer_pgr_results,
            weak_labels=weak_labels
        )

        print(f"\n✓ Results saved to: results/{exp_id}/")

    return {
        "all_results": all_results,
        "pgr_results": disclaimer_pgr_results,
        "baseline_pgr": baseline_pgr_metrics
    }


async def run_cot_sweep(
    model_pair: str = "8b_to_405b",
    k_values: List[int] = None,
    baseline_exp_path: str = None,
    weak_cot_labels_path: str = None,
    gold_cot_labels_path: str = None,
    save_results: bool = True
):
    """
    Run Chain-of-Thought experiments.

    Usage in Jupyter:
        await run_cot_sweep(
            "8b_to_405b",
            k_values=[0, 2, 5, 10],
            baseline_exp_path="results/8b_405b_baseline_20251102_133054/experiment.json",
            weak_cot_labels_path="data/cot_weak_labels/8b_cot_weak_labels_20251102_141149.json",
            gold_cot_labels_path="data/cot_gold_labels/405b_cot_gold_labels_20251102_141158.json"
        )

    Args:
        model_pair: One of "8b_to_405b", "8b_to_70b"
        k_values: List of K values
        baseline_exp_path: Path to baseline experiment
        weak_cot_labels_path: Path to weak CoT labels
        gold_cot_labels_path: Path to gold CoT labels
        save_results: Whether to save results

    Returns:
        Dictionary with results and comparison to baseline
    """
    if k_values is None:
        k_values = [0, 2, 5, 10]

    print("\n" + "="*70)
    print(f"CHAIN-OF-THOUGHT EXPERIMENT: {model_pair.upper()}")
    print("="*70)

    # Setup
    config = Config.from_env()
    config.setup_environment()

    pair = get_model_pair(model_pair)
    config.weak_model = pair.weak_model
    config.strong_model = pair.strong_model

    print(f"\nWeak Model:  {config.weak_model}")
    print(f"Strong Model: {config.strong_model}")
    print(f"Chain-of-Thought: ENABLED")

    # Load dataset
    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    # Load baseline
    if baseline_exp_path and Path(baseline_exp_path).exists():
        with open(baseline_exp_path) as f:
            baseline_data = json.load(f)
        baseline_pgr_metrics = {
            int(k): baseline_data['pgr_metrics'][k]
            for k in baseline_data['pgr_metrics'].keys()
        }
        print(f"✓ Loaded baseline PGR metrics")
    else:
        raise ValueError(f"Baseline not found: {baseline_exp_path}")

    # Load CoT labels
    if weak_cot_labels_path and Path(weak_cot_labels_path).exists():
        with open(weak_cot_labels_path) as f:
            weak_cot_data = json.load(f)
        weak_cot_labels = [ModelResponse(**wl) for wl in weak_cot_data['weak_labels']]
        print(f"✓ Loaded {len(weak_cot_labels)} weak CoT labels")
    else:
        raise ValueError(f"Weak CoT labels not found: {weak_cot_labels_path}")

    if gold_cot_labels_path and Path(gold_cot_labels_path).exists():
        with open(gold_cot_labels_path) as f:
            gold_cot_data = json.load(f)
        gold_cot_labels = [ModelResponse(**gl) for gl in gold_cot_data['gold_labels']]
        print(f"✓ Loaded {len(gold_cot_labels)} gold CoT labels")
    else:
        print("⚠ Gold CoT labels not found, will skip gold+CoT experiments")
        gold_cot_labels = None

    # Initialize runner WITH CoT
    runner = ExperimentRunner(config, use_cot=True)
    result_manager = ResultManager()

    # Run experiments
    all_results = {}
    cot_pgr_results = {}

    for k in k_values:
        print("\n" + "="*70)
        print(f"K = {k}")
        print("="*70)

        if k == 0:
            result = await runner.run_baseline(
                test_data=test_data,
                model_id=config.strong_model,
                experiment_name=f"strong_cot_k{k}"
            )
        else:
            result = await runner.run_few_shot_experiment(
                train_data=few_shot_pool[:k],
                test_data=test_data,
                model_id=config.strong_model,
                use_gold_labels=False,
                weak_responses=weak_cot_labels[:k],
                num_few_shot=k,
                experiment_name=f"strong_weak_cot_k{k}"
            )

        all_results[k] = {"strong_weak_cot": result}

        # Calculate PGR
        baseline_metrics = baseline_pgr_metrics[k]
        weak_baseline_acc = baseline_metrics['weak_baseline_accuracy']
        strong_gold_acc = baseline_metrics['strong_gold_accuracy']
        perf_gap = strong_gold_acc - weak_baseline_acc
        recovered = result.accuracy - weak_baseline_acc
        pgr = recovered / perf_gap if perf_gap != 0 else 1.0

        pgr_metrics = PGRMetrics(
            weak_baseline_accuracy=weak_baseline_acc,
            strong_gold_accuracy=strong_gold_acc,
            strong_weak_accuracy=result.accuracy,
            performance_gap=perf_gap,
            recovered_gap=recovered,
            pgr=pgr,
            pgr_percentage=f"{pgr*100:.1f}%"
        )
        cot_pgr_results[k] = pgr_metrics

        # Show comparison
        baseline_pgr = baseline_metrics['pgr']
        print(f"\nK={k} Comparison:")
        print(f"  Baseline PGR: {baseline_pgr:.3f}")
        print(f"  CoT PGR:      {pgr:.3f}")
        print(f"  Change:       {pgr - baseline_pgr:+.3f}")

    # Save if requested
    if save_results:
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
            k_values=k_values,
            temperature=config.temperature,
            seed=42
        )

        result_manager.save_experiment(
            experiment_id=exp_id,
            metadata=metadata,
            all_results=all_results,
            pgr_results=cot_pgr_results,
            weak_labels=weak_cot_labels
        )

        print(f"\n✓ Results saved to: results/{exp_id}/")

    return {
        "all_results": all_results,
        "pgr_results": cot_pgr_results,
        "baseline_pgr": baseline_pgr_metrics
    }
