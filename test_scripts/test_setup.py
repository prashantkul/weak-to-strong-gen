"""
Quick test to validate setup before running full experiments
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model_evaluator import ModelEvaluator, ModelResponse


async def test_api_connection():
    """Test that we can connect to OpenRouter API"""
    print("="*60)
    print("Testing API Connection")
    print("="*60)

    # Load config
    config = Config.from_env()
    config.setup_environment()

    print(f"✓ Configuration loaded")
    print(f"  Weak Model: {config.weak_model}")
    print(f"  Strong Model: {config.strong_model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Cache Dir: {config.cache_dir}")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    print(f"\n✓ ModelEvaluator initialized")

    # Test question
    test_question = """What is the capital of France?

A) London
B) Paris
C) Berlin
D) Madrid"""

    print(f"\nTest Question:\n{test_question}\n")

    # Test with weak model
    print(f"Querying {config.weak_model}...")
    result = await evaluator.evaluate_single(
        question=test_question,
        question_id=0,
        model_id=config.weak_model,
        verbose=True
    )

    print(f"\n✓ Response received:")
    print(f"  Answer: {result.answer}")
    print(f"  Raw response: {result.raw_response}")

    return True


async def test_few_shot():
    """Test few-shot prompting"""
    print("\n" + "="*60)
    print("Testing Few-Shot Prompting")
    print("="*60)

    config = Config.from_env()
    config.setup_environment()
    evaluator = ModelEvaluator(config)

    # Create simple few-shot examples
    few_shot_examples = [
        ("What is 2+2?\nA) 3\nB) 4\nC) 5", "B"),
        ("What color is the sky?\nA) Green\nB) Blue\nC) Red", "B"),
    ]

    few_shot_prompt = evaluator.create_few_shot_prompt(few_shot_examples)
    print(f"✓ Created few-shot prompt with {len(few_shot_examples)} examples")

    # Test question
    test_question = "What is the capital of Italy?\nA) Rome\nB) Milan\nC) Venice"

    print(f"\nTest with few-shot examples:")
    result = await evaluator.evaluate_single(
        question=test_question,
        question_id=1,
        model_id=config.weak_model,
        few_shot_prompt=few_shot_prompt,
        verbose=True
    )

    print(f"\n✓ Few-shot response:")
    print(f"  Answer: {result.answer}")
    print(f"  Raw: {result.raw_response}")

    return True


async def main():
    """Run all tests"""
    print("\nStarting Setup Tests\n")

    try:
        # Test 1: Basic API connection
        await test_api_connection()

        # Test 2: Few-shot prompting
        await test_few_shot()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nSetup is ready. You can now run the full notebook.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    asyncio.run(main())
