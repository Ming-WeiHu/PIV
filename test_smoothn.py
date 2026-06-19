"""Focused tests for the vendored Garcia smoothn port.

Verifies the three properties piv_simple relies on:
  1. noise reduction on a smooth field,
  2. missing-value (NaN) handling, and
  3. robust rejection of a gross outlier spike.
Run: python -m pytest test_smoothn.py -q   (or just `python test_smoothn.py`)
"""
import numpy as np

from smoothn import smoothn


def _field(n=64):
    """A smooth 2-D field (separable sine) to use as ground truth."""
    x = np.linspace(0, 2 * np.pi, n)
    gx, gy = np.meshgrid(x, x)
    return np.sin(gx) * np.cos(gy)


def test_reduces_noise():
    rng = np.random.RandomState(0)
    truth = _field()
    noisy = truth + 0.3 * rng.standard_normal(truth.shape)
    z = smoothn(noisy, isrobust=True)[0]
    err_noisy = np.sqrt(np.mean((noisy - truth) ** 2))
    err_smooth = np.sqrt(np.mean((z - truth) ** 2))
    assert err_smooth < err_noisy, (err_smooth, err_noisy)


def test_fills_nan():
    truth = _field()
    y = truth.copy()
    y[28:32, 28:32] = np.nan           # a small hole, as a masked PIV window would be
    z = smoothn(y, isrobust=True)[0]
    assert np.all(np.isfinite(z))      # hole is filled, no NaN left
    # filled values stay within the field's range (no blow-up) and track loosely
    assert z.min() >= truth.min() - 0.2 and z.max() <= truth.max() + 0.2
    assert np.sqrt(np.mean((z[28:32, 28:32] - truth[28:32, 28:32]) ** 2)) < 0.3


def test_robust_rejects_spike():
    truth = _field()
    y = truth.copy()
    y[32, 32] += 50.0                  # one gross spurious vector
    z = smoothn(y, isrobust=True)[0]
    # the spike must be pulled back toward the local field, not preserved
    assert abs(z[32, 32] - truth[32, 32]) < 1.0


if __name__ == "__main__":
    test_reduces_noise()
    test_fills_nan()
    test_robust_rejects_spike()
    print("all smoothn tests passed")
