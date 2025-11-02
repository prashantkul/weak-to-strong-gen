"""
Predefined model pairs for weak-to-strong experiments
"""

from typing import Dict, List
from pydantic import BaseModel


class ModelPair(BaseModel):
    """A weak-strong model pair"""
    name: str
    weak_model: str
    strong_model: str
    description: str
    parameter_ratio: float
    cost_multiplier: float  # Relative cost strong vs weak


# Predefined model pairs
LLAMA_8B_TO_70B = ModelPair(
    name="llama_8b_to_70b",
    weak_model="meta-llama/llama-3.1-8b-instruct",
    strong_model="meta-llama/llama-3.1-70b-instruct",
    description="Llama 3.1: 8B → 70B (8.75x parameters)",
    parameter_ratio=8.75,
    cost_multiplier=3.75  # 70B costs ~3.75x more than 8B
)

LLAMA_8B_TO_405B = ModelPair(
    name="llama_8b_to_405b",
    weak_model="meta-llama/llama-3.1-8b-instruct",
    strong_model="meta-llama/llama-3.1-405b-instruct",
    description="Llama 3.1: 8B → 405B (50x parameters)",
    parameter_ratio=50.0,
    cost_multiplier=15.0  # 405B costs ~15x more than 8B
)

LLAMA_70B_TO_405B = ModelPair(
    name="llama_70b_to_405b",
    weak_model="meta-llama/llama-3.1-70b-instruct",
    strong_model="meta-llama/llama-3.1-405b-instruct",
    description="Llama 3.1: 70B → 405B (5.8x parameters)",
    parameter_ratio=5.8,
    cost_multiplier=4.0  # 405B costs ~4x more than 70B
)


# Model pair registry
MODEL_PAIRS: Dict[str, ModelPair] = {
    "8b_to_70b": LLAMA_8B_TO_70B,
    "8b_to_405b": LLAMA_8B_TO_405B,
    "70b_to_405b": LLAMA_70B_TO_405B,
}


def get_model_pair(name: str) -> ModelPair:
    """Get model pair by name"""
    if name not in MODEL_PAIRS:
        raise ValueError(
            f"Unknown model pair: {name}. "
            f"Available: {list(MODEL_PAIRS.keys())}"
        )
    return MODEL_PAIRS[name]


def list_model_pairs() -> List[str]:
    """List available model pair names"""
    return list(MODEL_PAIRS.keys())


def print_model_pairs():
    """Print all available model pairs"""
    print("\nAvailable Model Pairs:")
    print("="*70)
    for name, pair in MODEL_PAIRS.items():
        print(f"\n{name}:")
        print(f"  {pair.description}")
        print(f"  Weak:  {pair.weak_model}")
        print(f"  Strong: {pair.strong_model}")
        print(f"  Ratio: {pair.parameter_ratio}x parameters")
        print(f"  Cost:  {pair.cost_multiplier}x API cost")
