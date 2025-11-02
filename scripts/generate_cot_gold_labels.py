"""
Generate Chain-of-Thought gold labels from 405B model
This creates few-shot examples with high-quality reasoning
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src import Config, DatasetManager, ModelEvaluator


async def main():
    print("\n" + "="*70)
    print("GENERATING CHAIN-OF-THOUGHT GOLD LABELS")
    print("405B Model with Reasoning")
    print("="*70)

    # Configuration
    config = Config.from_env()
    config.setup_environment()

    # Use 405B strong model
    pair = __import__('src').get_model_pair("8b_to_405b")
    strong_model_id = pair.strong_model

    print(f"\nStrong Model: {strong_model_id}")
    print(f"Temperature: {config.temperature} (deterministic)")
    print(f"Chain-of-Thought: ENABLED")

    # Load dataset
    print("\n" + "-"*70)
    print("LOADING DATASET")
    print("-"*70)

    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    print(f"✓ Few-shot pool: {len(few_shot_pool)} questions")

    # Generate for first 20 examples (same as weak labels)
    num_to_generate = 20
    questions_to_label = few_shot_pool[:num_to_generate]

    print(f"\nGenerating CoT gold labels for {num_to_generate} questions...")
    print(f"(Will use first 10 for K=10 experiments)")

    # Initialize evaluator WITH CoT
    evaluator = ModelEvaluator(config, use_cot=True)

    # Generate gold labels with reasoning
    print("\n" + "="*70)
    print("RUNNING 405B MODEL WITH CoT PROMPTING")
    print("="*70)

    questions = [(q.question_id, q.question) for q in questions_to_label]
    gold_cot_responses = await evaluator.evaluate_batch(
        questions=questions,
        model_id=strong_model_id,
        few_shot_prompt=None,  # No few-shot for generating gold labels
        verbose=True
    )

    # Show sample responses to verify reasoning quality
    print("\n" + "="*70)
    print("SAMPLE RESPONSES (first 2)")
    print("="*70)

    for i, resp in enumerate(gold_cot_responses[:2]):
        question = questions_to_label[i].question
        print(f"\n[Question {resp.question_id}]")
        print(f"Q: {question[:100]}...")
        print(f"\nRaw Response:")
        print(f"{resp.raw_response}")
        print(f"\nExtracted Answer: {resp.answer}")
        print("-"*70)

    # Calculate accuracy vs ground truth
    gt_map = {q.question_id: q.answer for q in questions_to_label}
    num_correct = sum(1 for r in gold_cot_responses if r.answer == gt_map[r.question_id])
    accuracy = num_correct / len(gold_cot_responses)

    print(f"\n405B Strong Model with CoT Accuracy: {accuracy:.2%} ({num_correct}/{len(gold_cot_responses)})")

    # Save results
    print("\n" + "="*70)
    print("SAVING COT GOLD LABELS")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/cot_gold_labels")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / f"405b_cot_gold_labels_{timestamp}.json"

    # Save as JSON
    data = {
        "metadata": {
            "timestamp": timestamp,
            "strong_model": strong_model_id,
            "temperature": config.temperature,
            "num_labels": len(gold_cot_responses),
            "accuracy": accuracy,
            "use_cot": True
        },
        "gold_labels": [r.model_dump() for r in gold_cot_responses]
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(gold_cot_responses)} CoT gold labels to: {output_file}")

    # Check reasoning quality
    has_reasoning = sum(1 for r in gold_cot_responses if len(r.raw_response.strip()) > 5)
    print(f"\nResponses with content (>5 chars): {has_reasoning}/{len(gold_cot_responses)}")

    # Check response lengths
    avg_length = sum(len(r.raw_response) for r in gold_cot_responses) / len(gold_cot_responses)
    print(f"Average response length: {avg_length:.1f} characters")

    # Compare with ground truth for sanity check
    print(f"\nGold CoT accuracy: {accuracy:.2%}")
    print(f"(Should be very high since 405B is strong model)")

    print("\n" + "="*70)
    print("✓ COT GOLD LABEL GENERATION COMPLETE")
    print("="*70)

    print(f"\nNext steps:")
    print(f"1. Review sample responses above to verify reasoning quality")
    print(f"2. Compare gold vs weak reasoning quality")
    print(f"3. Proceed with modifying few-shot formatting code")


if __name__ == "__main__":
    asyncio.run(main())
