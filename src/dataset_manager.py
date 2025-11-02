"""
Dataset management with proper train/test splits and persistence
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datasets import load_dataset
import pydantic
from abc import ABC, abstractmethod


class DatasetQuestion(pydantic.BaseModel):
    question_id: int
    question: str
    incorrect_answers: list[str]
    correct_answer: str
    solution: str


class FormattedDatasetQuestion(pydantic.BaseModel):
    question_id: int
    question: str
    answer: str
    solution: str


class DatasetSplit(pydantic.BaseModel):
    """Records exact train/test split for reproducibility"""
    train_ids: List[int]
    test_ids: List[int]
    few_shot_pool_ids: List[int]  # Subset of train for few-shot sampling
    seed: int
    dataset_name: str
    total_questions: int


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


class DatasetManager:
    """
    Manages dataset loading, splitting, and persistence
    Ensures no train/test contamination
    """

    def __init__(self, data_dir: Path = Path("./data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def create_and_save_split(
        self,
        test_size: int = 200,
        few_shot_pool_size: int = 100,
        seed: int = 42,
        dataset_name: str = "truthfulqa"
    ) -> Tuple[List[FormattedDatasetQuestion], List[FormattedDatasetQuestion], DatasetSplit]:
        """
        Create train/test split and save to disk

        Args:
            test_size: Number of questions for test set
            few_shot_pool_size: Number of training questions to reserve for few-shot sampling
            seed: Random seed
            dataset_name: Name for saving

        Returns:
            (test_data, few_shot_pool, split_metadata)
        """
        print(f"\nCreating dataset split (seed={seed})...")

        # Load dataset
        truthful_dataset = TruthfulDataset(dataset_split="validation")
        formatted_questions = truthful_dataset.format_dataset()

        total = len(formatted_questions)
        print(f"  Total questions: {total}")

        # Shuffle with fixed seed
        random.seed(seed)
        shuffled_indices = list(range(total))
        random.shuffle(shuffled_indices)

        # Split
        test_ids = shuffled_indices[:test_size]
        train_ids = shuffled_indices[test_size:]

        # Few-shot pool from training set
        few_shot_pool_ids = train_ids[:few_shot_pool_size]

        print(f"  Test set: {len(test_ids)} questions")
        print(f"  Train set: {len(train_ids)} questions")
        print(f"  Few-shot pool: {len(few_shot_pool_ids)} questions (subset of train)")

        # Verify no overlap
        assert len(set(test_ids) & set(few_shot_pool_ids)) == 0, "Test/train contamination!"

        # Create split metadata
        split = DatasetSplit(
            train_ids=train_ids,
            test_ids=test_ids,
            few_shot_pool_ids=few_shot_pool_ids,
            seed=seed,
            dataset_name=dataset_name,
            total_questions=total
        )

        # Save everything
        self._save_questions(formatted_questions, "all_questions.json")
        self._save_split(split, "split_metadata.json")

        # Extract actual question objects
        test_data = [formatted_questions[i] for i in test_ids]
        few_shot_pool = [formatted_questions[i] for i in few_shot_pool_ids]

        print(f"  ✓ Saved to {self.data_dir}/")

        return test_data, few_shot_pool, split

    def load_split(
        self,
        split_file: str = "split_metadata.json"
    ) -> Tuple[List[FormattedDatasetQuestion], List[FormattedDatasetQuestion], DatasetSplit]:
        """
        Load previously saved split

        Returns:
            (test_data, few_shot_pool, split_metadata)
        """
        print(f"\nLoading saved dataset split...")

        # Load metadata
        split = self._load_split(split_file)
        questions = self._load_questions("all_questions.json")

        # Extract sets
        test_data = [questions[i] for i in split.test_ids]
        few_shot_pool = [questions[i] for i in split.few_shot_pool_ids]

        print(f"  ✓ Loaded {len(test_data)} test, {len(few_shot_pool)} few-shot pool")
        print(f"  Seed: {split.seed}, Dataset: {split.dataset_name}")

        return test_data, few_shot_pool, split

    def _save_questions(self, questions: List[FormattedDatasetQuestion], filename: str):
        """Save questions to JSON"""
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump([q.model_dump() for q in questions], f, indent=2)

    def _load_questions(self, filename: str) -> List[FormattedDatasetQuestion]:
        """Load questions from JSON"""
        filepath = self.data_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [FormattedDatasetQuestion(**q) for q in data]

    def _save_split(self, split: DatasetSplit, filename: str):
        """Save split metadata to JSON"""
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(split.model_dump(), f, indent=2)

    def _load_split(self, filename: str) -> DatasetSplit:
        """Load split metadata from JSON"""
        filepath = self.data_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
        return DatasetSplit(**data)

    def split_exists(self, split_file: str = "split_metadata.json") -> bool:
        """Check if split file exists"""
        return (self.data_dir / split_file).exists()
