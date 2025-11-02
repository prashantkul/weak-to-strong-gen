"""
Mini experiment: Test with K={0, 2} on small dataset
"""

import asyncio
import sys
from pathlib import Path
from datasets import load_dataset
import random
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.experiment_runner import ExperimentRunner, FormattedDatasetQuestion
from src.results_analyzer import ResultsAnalyzer


# Dataset classes from notebook
import pydantic
from abc import ABC, abstractmethod


class DatasetQuestion(pydantic.BaseModel):
    question_id: int
    question: str
    incorrect_answers: list[str]
    correct_answer: str
    solution: str


class Dataset(ABC):
    def __init__(self, dataset: list[dict]):
        self.dataset = dataset

    @abstractmethod
    def unpack_single(self, row: dict, index: int) -> DatasetQuestion:
        pass

    def convert_to_questions(self, dataset: list[dict]) -> list[DatasetQuestion]:
        return [self.unpack_single(row, i) for i, row in enumerate(dataset)]

    def format_row(self, item: DatasetQuestion, seed: int = 42) -> FormattedDatasetQuestion:
        question_id = item.question_id
        question = item.question
        correct_answer = item.correct_answer
        incorrect_answers = item.incorrect_answers
        solution = item.solution
        assert correct_answer not in incorrect_answers, f"{correct_answer} in {incorrect_answers}"
        choices = [correct_answer] + incorrect_answers

        random.seed(question_id + seed)
        random.shuffle(choices)

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = letters[: len(choices)]
        choices_str = [f"{letters[i]}) {choice}" for i, choice in enumerate(choices)]
        choices_str = "\n".join(choices_str)
        correct_letter = letters[choices.index(correct_answer)]

        question = f"{question}\n\n{choices_str}"

        return FormattedDatasetQuestion(
            question_id=question_id,
            question=question,
            answer=correct_letter,
            solution=solution,
        )

    def format_dataset(self) -> list[FormattedDatasetQuestion]:
        return [self.format_row(q) for q in self.dataset]


class TruthfulDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice")
        hf_dataset = dataset[dataset_split]
        self.dataset = self.convert_to_questions(hf_dataset)

    @staticmethod
    def raw_to_question(raw):
        sentence = raw["question"].rstrip("\n")
        base_question = f"""{sentence}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)
        answer_key = item["label"]
        correct_answer = item["choices"][answer_key]

        incorrect_answers = [item["choices"][i] for i in range(len(item["choices"])) if i != answer_key]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
            solution=""
        )


async def main():
    print("\n" + "="*70)
    print("MINI EXPERIMENT: K={0, 2} with 20 test questions")
    print("="*70 + "\n")

    # Load config
    config = Config.from_env()
    config.setup_environment()

    print("Configuration:")
    print(f"  Weak:  {config.weak_model}")
    print(f"  Strong: {config.strong_model}")
    print(f"  Temperature: {config.temperature}")

    # Load dataset
    print("\nLoading TruthfulQA dataset...")
    truthful_dataset = TruthfulDataset(dataset_split="validation")
    formatted_truthful = truthful_dataset.format_dataset()

    # Use small subset for testing
    random.seed(42)
    truthful_all = random.sample(formatted_truthful, len(formatted_truthful))
    truthful_test = truthful_all[:20]  # Only 20 test questions
    truthful_train = truthful_all[20:]

    print(f"  Train: {len(truthful_train)} questions")
    print(f"  Test:  {len(truthful_test)} questions")

    # Initialize runner
    runner = ExperimentRunner(config)

    # K values
    K_VALUES = [0, 2]
    all_results = {}

    # Generate weak labels for training
    print("\n" + "-"*70)
    print("Generating weak labels for few-shot examples...")
    print("-"*70)
    train_pool = truthful_train[:10]
    train_questions = [(q.question_id, q.question) for q in train_pool]
    weak_labels = await runner.evaluator.generate_weak_labels(
        questions=train_questions,
        verbose=True
    )

    # Run experiments for each K
    for k in K_VALUES:
        print("\n" + "="*70)
        print(f"K = {k}")
        print("="*70)

        if k == 0:
            # Baseline
            weak_result = await runner.run_baseline(
                test_data=truthful_test,
                model_id=config.weak_model,
                experiment_name=f"weak_k{k}"
            )

            strong_result = await runner.run_baseline(
                test_data=truthful_test,
                model_id=config.strong_model,
                experiment_name=f"strong_k{k}"
            )

            all_results[k] = {
                "weak_baseline": weak_result,
                "strong_gold": strong_result,
                "strong_weak": strong_result
            }
        else:
            # Few-shot
            strong_gold = await runner.run_few_shot_experiment(
                train_data=train_pool[:k],
                test_data=truthful_test,
                model_id=config.strong_model,
                use_gold_labels=True,
                num_few_shot=k,
                experiment_name=f"strong_gold_k{k}"
            )

            strong_weak = await runner.run_few_shot_experiment(
                train_data=train_pool[:k],
                test_data=truthful_test,
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

    # Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")

    summary_data = []
    for k in K_VALUES:
        pgr = ResultsAnalyzer.analyze_experiment(all_results[k])
        summary_data.append({
            "K": k,
            "Weak": f"{pgr.weak_baseline_accuracy:.2%}",
            "Strong+Gold": f"{pgr.strong_gold_accuracy:.2%}",
            "Strong+Weak": f"{pgr.strong_weak_accuracy:.2%}",
            "PGR": f"{pgr.pgr:.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n" + "="*70)
    print("✓ MINI EXPERIMENT COMPLETED")
    print("="*70)
    print("\nReady to run full experiments in the notebook.")


if __name__ == "__main__":
    asyncio.run(main())
