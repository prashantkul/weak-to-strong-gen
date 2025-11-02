"""
Orchestrates weak-to-strong generalization experiments
"""

import random
from typing import List, Dict, Any
from pydantic import BaseModel

from .config import Config
from .model_evaluator import ModelEvaluator, ModelResponse


class FormattedDatasetQuestion(BaseModel):
    """Matches the format from the notebook"""
    question_id: int
    question: str
    answer: str
    solution: str


class ExperimentResults(BaseModel):
    """Results from a single experiment run"""
    experiment_name: str
    model_id: str
    prompt_type: str  # "gold", "weak", "none"
    num_few_shot: int
    responses: List[ModelResponse]
    ground_truth: List[str]
    accuracy: float
    num_correct: int
    num_total: int


class ExperimentRunner:
    """Orchestrates weak-to-strong ICL experiments"""

    def __init__(self, config: Config, use_disclaimer: bool = False, use_cot: bool = False):
        self.config = config
        self.use_disclaimer = use_disclaimer
        self.use_cot = use_cot
        self.evaluator = ModelEvaluator(config, use_disclaimer=use_disclaimer, use_cot=use_cot)

    def prepare_few_shot_examples(
        self,
        train_data: List[FormattedDatasetQuestion],
        weak_responses: List[ModelResponse] = None,
        gold_responses: List[ModelResponse] = None,
        use_gold_labels: bool = True,
        num_examples: int = None,
        seed: int = 42
    ) -> List[tuple]:
        """
        Prepare few-shot examples from training data

        Args:
            train_data: Training questions with gold answers
            weak_responses: Responses from weak model (if using weak labels)
            gold_responses: Responses from strong model (if using gold labels with CoT)
            use_gold_labels: If True, use gold answers; if False, use weak model answers
            num_examples: Number of examples to use (default: config.num_few_shot_examples)
            seed: Random seed for sampling

        Returns:
            List of tuples:
            - (question, answer) if no reasoning available
            - (question, answer, reasoning) if CoT responses provided
        """
        if num_examples is None:
            num_examples = self.config.num_few_shot_examples

        # Sample examples
        random.seed(seed)
        sampled_indices = random.sample(range(len(train_data)), num_examples)

        examples = []
        for idx in sampled_indices:
            question_data = train_data[idx]
            question = question_data.question

            if use_gold_labels:
                # Use ground truth answer
                answer = question_data.answer
                reasoning = None

                # If gold_responses provided (CoT), use reasoning from there
                if gold_responses and self.use_cot:
                    gold_resp = next(
                        (r for r in gold_responses if r.question_id == question_data.question_id),
                        None
                    )
                    if gold_resp and gold_resp.raw_response:
                        reasoning = gold_resp.raw_response
            else:
                # Use weak model's prediction
                if not weak_responses:
                    raise ValueError("weak_responses required when use_gold_labels=False")

                # Find the weak response for this question
                weak_resp = next(
                    (r for r in weak_responses if r.question_id == question_data.question_id),
                    None
                )
                if weak_resp:
                    answer = weak_resp.answer
                    # Extract reasoning if using CoT
                    reasoning = weak_resp.raw_response if (self.use_cot and weak_resp.raw_response) else None
                else:
                    # Fallback to gold if weak response not found
                    answer = question_data.answer
                    reasoning = None

            # Return 3-tuple if reasoning available, otherwise 2-tuple
            if reasoning:
                examples.append((question, answer, reasoning))
            else:
                examples.append((question, answer))

        return examples

    async def run_baseline(
        self,
        test_data: List[FormattedDatasetQuestion],
        model_id: str,
        experiment_name: str = "baseline"
    ) -> ExperimentResults:
        """
        Run baseline evaluation (no few-shot examples)

        Args:
            test_data: Test questions
            model_id: Model to evaluate
            experiment_name: Name for this experiment

        Returns:
            ExperimentResults
        """
        print(f"\n{'='*60}")
        print(f"Running baseline: {experiment_name}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        questions = [(q.question_id, q.question) for q in test_data]
        responses = await self.evaluator.evaluate_batch(
            questions=questions,
            model_id=model_id,
            few_shot_prompt=None,
            verbose=True
        )

        # Create ground truth mapping by question_id
        # (test_data is in random order, responses come back in any order from async)
        gt_map = {q.question_id: q.answer for q in test_data}

        # Calculate accuracy by matching question_id
        num_correct = 0
        for r in responses:
            if r.answer == gt_map[r.question_id]:
                num_correct += 1

        accuracy = num_correct / len(test_data) if len(test_data) > 0 else 0.0

        # Sort responses by question_id to maintain consistent ordering in results
        responses_sorted = sorted(responses, key=lambda r: r.question_id)
        ground_truth = [gt_map[r.question_id] for r in responses_sorted]

        print(f"\n✓ Accuracy: {accuracy:.2%} ({num_correct}/{len(ground_truth)})")

        return ExperimentResults(
            experiment_name=experiment_name,
            model_id=model_id,
            prompt_type="none",
            num_few_shot=0,
            responses=responses_sorted,  # Use sorted responses
            ground_truth=ground_truth,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=len(ground_truth)
        )

    async def run_few_shot_experiment(
        self,
        train_data: List[FormattedDatasetQuestion],
        test_data: List[FormattedDatasetQuestion],
        model_id: str,
        use_gold_labels: bool,
        weak_responses: List[ModelResponse] = None,
        gold_responses: List[ModelResponse] = None,
        num_few_shot: int = None,
        experiment_name: str = None
    ) -> ExperimentResults:
        """
        Run few-shot experiment

        Args:
            train_data: Training data for few-shot examples
            test_data: Test data for evaluation
            model_id: Model to evaluate
            use_gold_labels: Whether to use gold or weak labels in few-shot
            weak_responses: Weak model responses (required if use_gold_labels=False)
            gold_responses: Gold model responses (optional, for CoT with gold labels)
            num_few_shot: Number of few-shot examples
            experiment_name: Name for this experiment

        Returns:
            ExperimentResults
        """
        num_few_shot = num_few_shot or self.config.num_few_shot_examples
        prompt_type = "gold" if use_gold_labels else "weak"

        if experiment_name is None:
            experiment_name = f"{model_id}_{prompt_type}_{num_few_shot}shot"

        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Model: {model_id}")
        print(f"Few-shot: {num_few_shot} examples ({prompt_type} labels)")
        print(f"{'='*60}")

        # Prepare few-shot examples
        examples = self.prepare_few_shot_examples(
            train_data=train_data,
            weak_responses=weak_responses,
            gold_responses=gold_responses,
            use_gold_labels=use_gold_labels,
            num_examples=num_few_shot
        )

        few_shot_prompt = self.evaluator.create_few_shot_prompt(examples)

        # Evaluate on test set
        questions = [(q.question_id, q.question) for q in test_data]
        responses = await self.evaluator.evaluate_batch(
            questions=questions,
            model_id=model_id,
            few_shot_prompt=few_shot_prompt,
            verbose=True
        )

        # Create ground truth mapping by question_id
        # (test_data is in random order, responses come back in any order from async)
        gt_map = {q.question_id: q.answer for q in test_data}

        # Calculate accuracy by matching question_id
        num_correct = 0
        for r in responses:
            if r.answer == gt_map[r.question_id]:
                num_correct += 1

        accuracy = num_correct / len(test_data) if len(test_data) > 0 else 0.0

        # Sort responses by question_id to maintain consistent ordering in results
        responses_sorted = sorted(responses, key=lambda r: r.question_id)
        ground_truth = [gt_map[r.question_id] for r in responses_sorted]

        print(f"\n✓ Accuracy: {accuracy:.2%} ({num_correct}/{len(ground_truth)})")

        return ExperimentResults(
            experiment_name=experiment_name,
            model_id=model_id,
            prompt_type=prompt_type,
            num_few_shot=num_few_shot,
            responses=responses_sorted,  # Use sorted responses
            ground_truth=ground_truth,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=len(ground_truth)
        )

    async def run_full_experiment(
        self,
        train_data: List[FormattedDatasetQuestion],
        test_data: List[FormattedDatasetQuestion],
        num_few_shot: int = None
    ) -> Dict[str, ExperimentResults]:
        """
        Run complete weak-to-strong experiment:
        1. Generate weak labels from train data
        2. Evaluate weak model baseline
        3. Evaluate strong model with gold labels
        4. Evaluate strong model with weak labels

        Args:
            train_data: Training data
            test_data: Test data
            num_few_shot: Number of few-shot examples

        Returns:
            Dictionary of experiment results
        """
        num_few_shot = num_few_shot or self.config.num_few_shot_examples
        results = {}

        # Step 1: Generate weak labels on training data
        print(f"\n{'#'*60}")
        print("STEP 1: Generating weak model labels on training data")
        print(f"{'#'*60}")

        # We only need labels for the few-shot examples we'll use
        train_questions = [(q.question_id, q.question) for q in train_data[:num_few_shot * 2]]
        weak_train_responses = await self.evaluator.generate_weak_labels(
            questions=train_questions,
            verbose=True
        )

        # Step 2: Weak model baseline on test set
        print(f"\n{'#'*60}")
        print("STEP 2: Weak model baseline")
        print(f"{'#'*60}")

        results["weak_baseline"] = await self.run_baseline(
            test_data=test_data,
            model_id=self.config.weak_model,
            experiment_name="weak_baseline"
        )

        # Step 3: Strong model with gold labels (ceiling)
        print(f"\n{'#'*60}")
        print("STEP 3: Strong model with GOLD labels")
        print(f"{'#'*60}")

        results["strong_gold"] = await self.run_few_shot_experiment(
            train_data=train_data,
            test_data=test_data,
            model_id=self.config.strong_model,
            use_gold_labels=True,
            num_few_shot=num_few_shot,
            experiment_name="strong_gold"
        )

        # Step 4: Strong model with weak labels (main experiment)
        print(f"\n{'#'*60}")
        print("STEP 4: Strong model with WEAK labels")
        print(f"{'#'*60}")

        results["strong_weak"] = await self.run_few_shot_experiment(
            train_data=train_data,
            test_data=test_data,
            model_id=self.config.strong_model,
            use_gold_labels=False,
            weak_responses=weak_train_responses,
            num_few_shot=num_few_shot,
            experiment_name="strong_weak"
        )

        return results
