import pytest
from train_model import discretize_return


class TestDiscreetizeReturn:
    def test_strong_up(self):
        assert discretize_return(0.06) == 2

    def test_mild_up(self):
        assert discretize_return(0.04) == 1

    def test_exact_5pct_is_mild_up(self):
        # +5% is the upper boundary of class 1 (UP 3-5%)
        assert discretize_return(0.05) == 1

    def test_exact_3pct_is_stable(self):
        # +3% is the upper boundary of class 0 (STABLE)
        assert discretize_return(0.03) == 0

    def test_stable_zero(self):
        assert discretize_return(0.0) == 0

    def test_stable_small_positive(self):
        assert discretize_return(0.02) == 0

    def test_stable_small_negative(self):
        assert discretize_return(-0.02) == 0

    def test_exact_neg_3pct_is_stable(self):
        # -3% is the lower boundary of class 0 (STABLE)
        assert discretize_return(-0.03) == 0

    def test_mild_down(self):
        assert discretize_return(-0.04) == -1

    def test_exact_neg_5pct_is_mild_down(self):
        # -5% is the upper boundary of class -1 (DOWN 3-5%)
        assert discretize_return(-0.05) == -1

    def test_strong_down(self):
        assert discretize_return(-0.06) == -2
