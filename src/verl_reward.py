"""Math reward for veRL rollouts and validation.

Training rewards and evaluation accuracy use the same boxed-answer scorer.
"""

from typing import Optional

from src.opsd_format import (
    extract_boxed_answer,
    grade_boxed_answer,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Score a completion by its final boxed answer."""
    del data_source, extra_info, kwargs
    predicted = extract_boxed_answer(solution_str)
    correct = grade_boxed_answer(predicted, ground_truth) if predicted is not None else False
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "formatted": 1.0 if predicted is not None else 0.0,
    }
