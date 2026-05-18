"""MODEL_WEIGHTS entries must sum to 1.0 for interpretable diagnostic ensembles."""

import pytest

from src.baseline_runner import MODEL_WEIGHTS


@pytest.mark.parametrize("name", sorted(MODEL_WEIGHTS))
def test_diagnostic_weight_vectors_sum_to_one(name: str) -> None:
    total = sum(MODEL_WEIGHTS[name].values())
    assert abs(total - 1.0) < 1e-9, f"{name}: weights sum to {total}, expected 1.0"
