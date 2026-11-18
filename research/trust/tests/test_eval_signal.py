import numpy as np
import pytest

from research.trust.eval_signal import signal_error_auc


def test_perfect_ranking():
    result = signal_error_auc(
        [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_boot=50
    )
    assert result == pytest.approx((1.0, 1.0, 1.0))


def test_reversed_ranking():
    result = signal_error_auc(
        [0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1], n_boot=50
    )
    assert result == pytest.approx((0.0, 0.0, 0.0))


def test_tied_scores():
    result = signal_error_auc(
        [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1], n_boot=50
    )
    assert result == pytest.approx((0.5, 0.5, 0.5))


@pytest.mark.parametrize("wrong", [[0, 0], [1, 1]])
def test_single_class(wrong):
    assert signal_error_auc([0.2, 0.8], wrong) == (None, None, None)


@pytest.mark.parametrize(
    "signal, wrong",
    [
        ([], []),
        ([0.1, 0.2], [0]),
        ([[0.1, 0.2]], [[0, 1]]),
        ([np.nan, 0.2], [0, 1]),
        ([np.inf, 0.2], [0, 1]),
        ([0.1, 0.2], [0, 0.5]),
        ([0.1, 0.2], [0, 2]),
        ([0.1, 0.2], [0, np.nan]),
    ],
)
def test_invalid_inputs(signal, wrong):
    with pytest.raises(ValueError):
        signal_error_auc(signal, wrong)


@pytest.mark.parametrize("n_boot", [0, -1, 1.5, True])
def test_invalid_bootstrap_count(n_boot):
    with pytest.raises(ValueError):
        signal_error_auc([0.1, 0.9], [0, 1], n_boot=n_boot)


def test_reproducibility():
    signal = [0.1, 0.6, 0.3, 0.8, 0.5, 0.9]
    wrong = [0, 0, 1, 1, 0, 1]
    first = signal_error_auc(signal, wrong, n_boot=100, seed=42)
    second = signal_error_auc(signal, wrong, n_boot=100, seed=42)

    assert first == second
    auc, lo, hi = first
    assert 0 <= auc <= 1
    assert 0 <= lo <= hi <= 1