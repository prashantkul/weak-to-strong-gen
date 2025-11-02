"""
Quick test to verify notebook functions work correctly
Run this in a Jupyter cell to test the setup
"""

async def test_notebook_setup():
    """
    Quick test of notebook functions

    Usage in Jupyter:
        await test_notebook_setup()
    """
    print("Testing notebook experiment functions...")
    print("="*70)

    # Test imports
    try:
        from notebook_experiments import (
            run_baseline_sweep,
            run_disclaimer_sweep,
            run_cot_sweep
        )
        print("✓ Notebook functions imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test src imports
    try:
        from src import Config, DatasetManager, ExperimentRunner
        print("✓ Source modules imported successfully")
    except Exception as e:
        print(f"✗ Source import failed: {e}")
        return False

    # Test config
    try:
        config = Config.from_env()
        print(f"✓ Config loaded successfully")
        print(f"  - Weak model: {config.weak_model}")
        print(f"  - Strong model: {config.strong_model}")
    except Exception as e:
        print(f"✗ Config failed: {e}")
        return False

    # Test dataset loading
    try:
        dm = DatasetManager()
        test_data, few_shot_pool, split = dm.load_split()
        print(f"✓ Dataset loaded successfully")
        print(f"  - Test set: {len(test_data)} questions")
        print(f"  - Few-shot pool: {len(few_shot_pool)} questions")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

    # Test a small API call
    try:
        from src.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator(config)

        # Test with just 1 question
        test_question = test_data[0]
        response = await evaluator.evaluate_single(
            question=test_question.question,
            question_id=test_question.question_id,
            model_id=config.weak_model,
            few_shot_prompt=None,
            verbose=False
        )

        print(f"✓ API call successful")
        print(f"  - Question ID: {response.question_id}")
        print(f"  - Answer: {response.answer}")
        print(f"  - Model: {response.model_id}")

    except Exception as e:
        print(f"✗ API call failed: {e}")
        return False

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - Ready to run experiments!")
    print("="*70)

    return True


# Usage instructions
if __name__ == "__main__":
    print("""
    To run this test in a Jupyter notebook, use:

        await test_notebook_setup()

    This will verify:
    1. All imports work
    2. Configuration loads correctly
    3. Dataset is accessible
    4. API calls work

    If all tests pass, you're ready to run the full experiments!
    """)
