"""
Generate Chain-of-Thought weak labels from 8B model
This creates few-shot examples with reasoning for proper CoT experiments
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
    print("GENERATING CHAIN-OF-THOUGHT WEAK LABELS")
    print("8B Model with Reasoning")
    print("="*70)

    # Configuration
    config = Config.from_env()
    config.setup_environment()

    # Use 8B weak model
    pair_8b_405b = __import__('src').get_model_pair("8b_to_405b")
    weak_model_id = pair_8b_405b.weak_model

    print(f"\nWeak Model: {weak_model_id}")
    print(f"Temperature: {config.temperature} (deterministic)")
    print(f"Chain-of-Thought: ENABLED")

    # Load dataset
    print("\n" + "-"*70)
    print("LOADING DATASET")
    print("-"*70)

    dm = DatasetManager()
    test_data, few_shot_pool, split = dm.load_split()

    print(f"✓ Few-shot pool: {len(few_shot_pool)} questions")

    # We need weak labels for K=10, so generate for first 10 examples
    # (Actually generate for first 20 to have extras if needed)
    num_to_generate = 20
    questions_to_label = few_shot_pool[:num_to_generate]

    print(f"\nGenerating CoT weak labels for {num_to_generate} questions...")
    print(f"(Will use first 10 for K=10 experiments)")

    # Initialize evaluator WITH CoT
    evaluator = ModelEvaluator(config, use_cot=True)

    # Generate weak labels with reasoning
    print("\n" + "="*70)
    print("RUNNING 8B MODEL WITH CoT PROMPTING")
    print("="*70)

    questions = [(q.question_id, q.question) for q in questions_to_label]
    weak_cot_responses = await evaluator.evaluate_batch(
        questions=questions,
        model_id=weak_model_id,
        few_shot_prompt=None,  # No few-shot for generating weak labels
        verbose=True
    )

    # Show sample responses to verify reasoning was captured
    print("\n" + "="*70)
    print("SAMPLE RESPONSES (first 2)")
    print("="*70)

    for i, resp in enumerate(weak_cot_responses[:2]):
        question = questions_to_label[i].question
        print(f"\n[Question {resp.question_id}]")
        print(f"Q: {question[:100]}...")
        print(f"\nRaw Response:")
        print(f"{resp.raw_response}")
        print(f"\nExtracted Answer: {resp.answer}")
        print("-"*70)

    # Calculate accuracy vs ground truth
    gt_map = {q.question_id: q.answer for q in questions_to_label}
    num_correct = sum(1 for r in weak_cot_responses if r.answer == gt_map[r.question_id])
    accuracy = num_correct / len(weak_cot_responses)

    print(f"\n8B Weak Model with CoT Accuracy: {accuracy:.2%} ({num_correct}/{len(weak_cot_responses)})")

    # Save results
    print("\n" + "="*70)
    print("SAVING COT WEAK LABELS")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/cot_weak_labels")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / f"8b_cot_weak_labels_{timestamp}.json"

    # Save as JSON
    data = {
        "metadata": {
            "timestamp": timestamp,
            "weak_model": weak_model_id,
            "temperature": config.temperature,
            "num_labels": len(weak_cot_responses),
            "accuracy": accuracy,
            "use_cot": True
        },
        "weak_labels": [r.model_dump() for r in weak_cot_responses]
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(weak_cot_responses)} CoT weak labels to: {output_file}")

    # Also check if any responses have actual reasoning content
    has_reasoning = sum(1 for r in weak_cot_responses if len(r.raw_response.strip()) > 5)
    print(f"\nResponses with content (>5 chars): {has_reasoning}/{len(weak_cot_responses)}")

    # Check response lengths
    avg_length = sum(len(r.raw_response) for r in weak_cot_responses) / len(weak_cot_responses)
    print(f"Average response length: {avg_length:.1f} characters")

    print("\n" + "="*70)
    print("✓ COT WEAK LABEL GENERATION COMPLETE")
    print("="*70)

    print(f"\nNext steps:")
    print(f"1. Review sample responses above to verify reasoning quality")
    print(f"2. For gold reasoning: Need to obtain/generate gold explanations")
    print(f"3. Modify prepare_few_shot_examples() to include reasoning field")
    print(f"4. Update create_few_shot_prompt() to format with reasoning")
    print(f"5. Then run CoT experiments with proper reasoning examples")

    print(f"\nNote: Current CoT scripts use prompt-only approach (no reasoning in examples)")
    print(f"      For full CoT, need to integrate these reasoning-enhanced labels")


if __name__ == "__main__":
    asyncio.run(main())
