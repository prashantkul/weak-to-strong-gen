"""
Analysis and visualization of experiment results
"""

import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel

from .experiment_runner import ExperimentResults


class PGRMetrics(BaseModel):
    """Performance Gap Recovered metrics"""
    weak_baseline_accuracy: float
    strong_gold_accuracy: float
    strong_weak_accuracy: float
    performance_gap: float  # strong_gold - weak_baseline
    recovered_gap: float  # strong_weak - weak_baseline
    pgr: float  # recovered_gap / performance_gap
    pgr_percentage: str

    def __str__(self):
        return f"""
Performance Gap Recovered (PGR) Metrics:
{'='*50}
Weak Model Baseline:      {self.weak_baseline_accuracy:.2%}
Strong Model (Gold):      {self.strong_gold_accuracy:.2%}
Strong Model (Weak):      {self.strong_weak_accuracy:.2%}

Performance Gap:          {self.performance_gap:.2%}
Recovered Gap:            {self.recovered_gap:.2%}
PGR:                      {self.pgr:.3f} ({self.pgr_percentage})
{'='*50}
"""


class ResultsAnalyzer:
    """Analyzes experiment results and calculates PGR"""

    @staticmethod
    def calculate_pgr(
        weak_baseline_acc: float,
        strong_gold_acc: float,
        strong_weak_acc: float
    ) -> PGRMetrics:
        """
        Calculate Performance Gap Recovered (PGR) metric

        PGR = (Perf_strong_weak - Perf_weak) / (Perf_strong_gold - Perf_weak)

        Args:
            weak_baseline_acc: Weak model accuracy
            strong_gold_acc: Strong model accuracy with gold labels
            strong_weak_acc: Strong model accuracy with weak labels

        Returns:
            PGRMetrics object
        """
        performance_gap = strong_gold_acc - weak_baseline_acc
        recovered_gap = strong_weak_acc - weak_baseline_acc

        if performance_gap == 0:
            pgr = 1.0 if recovered_gap == 0 else float('inf')
        else:
            pgr = recovered_gap / performance_gap

        pgr_percentage = f"{pgr * 100:.1f}%"

        return PGRMetrics(
            weak_baseline_accuracy=weak_baseline_acc,
            strong_gold_accuracy=strong_gold_acc,
            strong_weak_accuracy=strong_weak_acc,
            performance_gap=performance_gap,
            recovered_gap=recovered_gap,
            pgr=pgr,
            pgr_percentage=pgr_percentage
        )

    @staticmethod
    def analyze_experiment(results: Dict[str, ExperimentResults]) -> PGRMetrics:
        """
        Analyze full experiment results and calculate PGR

        Args:
            results: Dictionary with keys 'weak_baseline', 'strong_gold', 'strong_weak'

        Returns:
            PGRMetrics object
        """
        weak_acc = results["weak_baseline"].accuracy
        strong_gold_acc = results["strong_gold"].accuracy
        strong_weak_acc = results["strong_weak"].accuracy

        return ResultsAnalyzer.calculate_pgr(weak_acc, strong_gold_acc, strong_weak_acc)

    @staticmethod
    def create_results_dataframe(results: Dict[str, ExperimentResults]) -> pd.DataFrame:
        """
        Create a DataFrame summarizing all experiment results

        Args:
            results: Dictionary of experiment results

        Returns:
            pandas DataFrame
        """
        data = []
        for name, result in results.items():
            data.append({
                "Experiment": name,
                "Model": result.model_id.split("/")[-1],  # Short model name
                "Prompt Type": result.prompt_type,
                "Few-Shot": result.num_few_shot,
                "Accuracy": f"{result.accuracy:.2%}",
                "Correct": result.num_correct,
                "Total": result.num_total
            })

        return pd.DataFrame(data)

    @staticmethod
    def analyze_errors(
        result: ExperimentResults,
        show_top_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze which questions were answered incorrectly

        Args:
            result: ExperimentResults to analyze
            show_top_n: Number of errors to show

        Returns:
            DataFrame of errors
        """
        errors = []
        for response, gt in zip(result.responses, result.ground_truth):
            if response.answer != gt:
                errors.append({
                    "Question ID": response.question_id,
                    "Predicted": response.answer,
                    "Correct": gt,
                    "Reasoning": response.reasoning[:100] + "..." if len(response.reasoning) > 100 else response.reasoning
                })

        df = pd.DataFrame(errors)
        return df.head(show_top_n) if len(df) > 0 else df

    @staticmethod
    def compare_predictions(
        result1: ExperimentResults,
        result2: ExperimentResults,
        name1: str = "Model 1",
        name2: str = "Model 2"
    ) -> Dict[str, List[int]]:
        """
        Compare predictions between two experiment results

        Args:
            result1: First experiment results
            result2: Second experiment results
            name1: Name for first experiment
            name2: Name for second experiment

        Returns:
            Dictionary with question IDs for different agreement categories
        """
        both_correct = []
        both_wrong = []
        only_1_correct = []
        only_2_correct = []

        for r1, r2, gt in zip(result1.responses, result2.responses, result1.ground_truth):
            q_id = r1.question_id
            r1_correct = (r1.answer == gt)
            r2_correct = (r2.answer == gt)

            if r1_correct and r2_correct:
                both_correct.append(q_id)
            elif not r1_correct and not r2_correct:
                both_wrong.append(q_id)
            elif r1_correct:
                only_1_correct.append(q_id)
            else:
                only_2_correct.append(q_id)

        print(f"\nPrediction Comparison: {name1} vs {name2}")
        print(f"{'='*50}")
        print(f"Both correct:        {len(both_correct)}")
        print(f"Both wrong:          {len(both_wrong)}")
        print(f"Only {name1} correct: {len(only_1_correct)}")
        print(f"Only {name2} correct: {len(only_2_correct)}")

        return {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            f"only_{name1}_correct": only_1_correct,
            f"only_{name2}_correct": only_2_correct
        }

    @staticmethod
    def print_summary(results: Dict[str, ExperimentResults], pgr_metrics: PGRMetrics):
        """
        Print a comprehensive summary of all results

        Args:
            results: Dictionary of experiment results
            pgr_metrics: PGR metrics
        """
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        df = ResultsAnalyzer.create_results_dataframe(results)
        print("\n" + df.to_string(index=False))

        print(pgr_metrics)

        # Interpretation
        print("\nInterpretation:")
        print("-" * 50)
        if pgr_metrics.pgr >= 0.9:
            print("✓ Excellent recovery! Strong model nearly fully recovers despite weak labels.")
        elif pgr_metrics.pgr >= 0.5:
            print("✓ Good recovery. Strong model shows robustness to weak labels.")
        elif pgr_metrics.pgr >= 0.0:
            print("⚠ Partial recovery. Strong model improves over weak baseline but limited.")
        else:
            print("✗ Negative PGR. Strong model performs worse with weak labels than weak baseline.")
