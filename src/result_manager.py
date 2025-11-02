"""
Experiment result persistence and tracking
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel

from .experiment_runner import ExperimentResults
from .model_evaluator import ModelResponse
from .results_analyzer import PGRMetrics


class ExperimentMetadata(BaseModel):
    """Metadata about an experiment run"""
    experiment_id: str
    timestamp: str
    weak_model: str
    strong_model: str
    dataset_name: str
    test_size: int
    k_values: List[int]
    temperature: float
    seed: int
    total_api_calls: Optional[int] = None
    estimated_cost: Optional[float] = None


class ExperimentArtifacts(BaseModel):
    """All artifacts from an experiment"""
    metadata: ExperimentMetadata
    results: Dict[int, Dict[str, Any]]  # K -> {weak_baseline, strong_gold, strong_weak}
    pgr_metrics: Dict[int, Dict[str, Any]]  # K -> PGR metrics
    weak_labels_cache: Optional[List[Dict[str, Any]]] = None  # Cached weak predictions


class ResultManager:
    """Manages saving and loading experiment results"""

    def __init__(self, results_dir: Path = Path("./results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_id(
        self,
        weak_model: str,
        strong_model: str,
        variant: str = "baseline",
        timestamp: Optional[str] = None
    ) -> str:
        """
        Create unique experiment ID with consistent naming

        Format: {weak}_{strong}_{variant}_{timestamp}
        Example: 8b_405b_baseline_20251102_130630
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model sizes (e.g., "8b", "70b", "405b")
        def extract_size(model_name: str) -> str:
            # Extract from patterns like "llama-3.1-8b-instruct"
            parts = model_name.lower().split("-")
            for part in parts:
                if part.endswith("b") and part[:-1].replace(".", "").isdigit():
                    return part
            return "unknown"

        weak_size = extract_size(weak_model)
        strong_size = extract_size(strong_model)

        return f"{weak_size}_{strong_size}_{variant}_{timestamp}"

    def save_experiment(
        self,
        experiment_id: str,
        metadata: ExperimentMetadata,
        all_results: Dict[int, Dict[str, ExperimentResults]],
        pgr_results: Dict[int, PGRMetrics],
        weak_labels: Optional[List[ModelResponse]] = None
    ):
        """
        Save complete experiment to disk

        Args:
            experiment_id: Unique identifier
            metadata: Experiment metadata
            all_results: Results for each K value
            pgr_results: PGR metrics for each K
            weak_labels: Optional cached weak model predictions
        """
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving experiment results to {exp_dir}/")

        # Convert results to serializable format
        serialized_results = {}
        for k, results_dict in all_results.items():
            serialized_results[k] = {}
            for name, result in results_dict.items():
                serialized_results[k][name] = {
                    "experiment_name": result.experiment_name,
                    "model_id": result.model_id,
                    "prompt_type": result.prompt_type,
                    "num_few_shot": result.num_few_shot,
                    "accuracy": result.accuracy,
                    "num_correct": result.num_correct,
                    "num_total": result.num_total,
                    "responses": [r.model_dump() for r in result.responses],
                    "ground_truth": result.ground_truth
                }

        # Convert PGR results
        serialized_pgr = {
            k: pgr.model_dump() for k, pgr in pgr_results.items()
        }

        # Convert weak labels if provided
        serialized_weak_labels = None
        if weak_labels:
            serialized_weak_labels = [wl.model_dump() for wl in weak_labels]

        # Create artifacts
        artifacts = ExperimentArtifacts(
            metadata=metadata,
            results=serialized_results,
            pgr_metrics=serialized_pgr,
            weak_labels_cache=serialized_weak_labels
        )

        # Save main results file
        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(artifacts.model_dump(), f, indent=2)

        # Save summary for quick viewing
        self._save_summary(exp_dir, metadata, pgr_results)

        print(f"  ✓ Saved experiment.json")
        print(f"  ✓ Saved summary.txt")

    def _save_summary(
        self,
        exp_dir: Path,
        metadata: ExperimentMetadata,
        pgr_results: Dict[int, PGRMetrics]
    ):
        """Save human-readable summary"""
        summary_path = exp_dir / "summary.txt"

        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write(f"Experiment ID: {metadata.experiment_id}\n")
            f.write(f"Timestamp: {metadata.timestamp}\n")
            f.write(f"Weak Model: {metadata.weak_model}\n")
            f.write(f"Strong Model: {metadata.strong_model}\n")
            f.write(f"Test Size: {metadata.test_size}\n")
            f.write(f"K Values: {metadata.k_values}\n")
            f.write(f"Temperature: {metadata.temperature}\n")
            f.write(f"Seed: {metadata.seed}\n\n")

            f.write("="*70 + "\n")
            f.write("RESULTS BY K\n")
            f.write("="*70 + "\n\n")

            for k in sorted(pgr_results.keys()):
                pgr = pgr_results[k]
                f.write(f"K = {k}\n")
                f.write(f"  Weak Baseline:  {pgr.weak_baseline_accuracy:.2%}\n")
                f.write(f"  Strong + Gold:  {pgr.strong_gold_accuracy:.2%}\n")
                f.write(f"  Strong + Weak:  {pgr.strong_weak_accuracy:.2%}\n")
                f.write(f"  PGR:            {pgr.pgr:.3f} ({pgr.pgr_percentage})\n\n")

    def load_experiment(self, experiment_id: str) -> ExperimentArtifacts:
        """Load saved experiment"""
        exp_dir = self.results_dir / experiment_id
        artifacts_path = exp_dir / "experiment.json"

        if not artifacts_path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")

        with open(artifacts_path, 'r') as f:
            data = json.load(f)

        return ExperimentArtifacts(**data)

    def list_experiments(self) -> List[str]:
        """List all saved experiments"""
        if not self.results_dir.exists():
            return []

        experiments = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and (item / "experiment.json").exists():
                experiments.append(item.name)

        return sorted(experiments)

    def save_weak_labels(
        self,
        weak_model: str,
        weak_labels: List[ModelResponse],
        question_ids: List[int],
        timestamp: Optional[str] = None
    ):
        """
        Save weak model predictions for reuse

        Args:
            weak_model: Model ID
            weak_labels: Predictions
            question_ids: Which questions were labeled
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        weak_short = weak_model.split("/")[-1].replace("-", "")
        filename = f"weak_labels_{weak_short}_{timestamp}.json"

        data = {
            "model": weak_model,
            "timestamp": timestamp,
            "question_ids": question_ids,
            "predictions": [wl.model_dump() for wl in weak_labels]
        }

        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  ✓ Saved weak labels to {filename}")

        return filename

    def load_weak_labels(self, filename: str) -> Tuple[List[ModelResponse], List[int]]:
        """Load previously saved weak labels"""
        filepath = self.results_dir / filename

        with open(filepath, 'r') as f:
            data = json.load(f)

        predictions = [ModelResponse(**p) for p in data["predictions"]]
        question_ids = data["question_ids"]

        return predictions, question_ids
