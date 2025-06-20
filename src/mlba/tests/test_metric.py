"""
Utility functions for

Machine Learning for Business Analytics:
Concepts, Techniques, and Applications in Python

(c) 2019-2025 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""
import unittest
from collections import namedtuple
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd
from sklearn.metrics import r2_score

from mlba import adjusted_r2_score, AIC_score, BIC_score
from mlba import regressionSummary, classificationSummary
import pytest

MockModel = namedtuple('MockModel', 'coef_')


class TestMetric(unittest.TestCase):
    def test_adjusted_r2_score(self) -> None:
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 3, 2, 5, 4]

        r2 = r2_score(y_true, y_pred)
        assert r2 == 0.6

        for df in range(4):
            n = len(y_true)
            f = (n - 1) / (n - df - 1)
            expected = 1 - (1 - r2) * f
            assert (adjusted_r2_score(y_true, y_pred, MockModel(coef_=[1] * df)) ==
                    pytest.approx(expected, abs=1e-3)), f'failed for df={df}'

        # if degree of freedom gets too large returns 0
        coef = [1] * 4
        assert (adjusted_r2_score(y_true, y_pred, MockModel(coef_=coef)) ==
                pytest.approx(0, abs=1e-3))

        coef = [1] * 5
        assert (adjusted_r2_score(y_true, y_pred, MockModel(coef_=coef)) ==
                pytest.approx(0, abs=1e-3))

    def test_AIC_score(self) -> None:
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 3, 2, 5, 4]

        assert (AIC_score(y_true=y_true, y_pred=y_pred, model=MockModel(coef_=[1] * 2)) ==
                pytest.approx(21.0736, abs=1e-3))

        assert (AIC_score(y_true=y_true, y_pred=y_pred, df=3) ==
                pytest.approx(AIC_score(y_true=y_true, y_pred=y_pred, model=MockModel(coef_=[1] * 2)), abs=1e-3))

        assert (AIC_score(y_true=y_true, y_pred=y_pred, df=3) >
                AIC_score(y_true=y_true, y_pred=y_pred, df=2))

    def test_AIC_score_classification(self) -> None:
        y_true = ['b' if y == 1 else 'a' for y in [1, 0, 0, 1, 1, 1]]
        y_pred = ['b' if y == 1 else 'a' for y in [1, 0, 1, 1, 0, 0]]
        assert (AIC_score(y_true=y_true, y_pred=y_pred, df=3) ==
                pytest.approx(20.868379, abs=1e-3))

        assert (AIC_score(y_true=pd.Series(y_true), y_pred=pd.Series(y_pred), df=3) ==
                pytest.approx(20.868379, abs=1e-3))

    def test_BIC_score(self) -> None:
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 3, 2, 5, 4]

        assert (BIC_score(y_true=y_true, y_pred=y_pred, model=MockModel(coef_=[1] * 2)) ==
                pytest.approx(19.51141, abs=1e-3))

        assert (BIC_score(y_true=y_true, y_pred=y_pred, df=3) ==
                pytest.approx(BIC_score(y_true=y_true, y_pred=y_pred, model=MockModel(coef_=[1] * 2)), rel=1e-3))

        assert BIC_score(y_true=y_true, y_pred=y_pred, df=3) > BIC_score(y_true=y_true, y_pred=y_pred, df=2)

    def test_regressionSummary(self) -> None:
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 3, 2, 5, 4]

        out = StringIO()
        with redirect_stdout(out):
            regressionSummary(y_true=y_true, y_pred=y_pred)
        s = out.getvalue()
        assert 'Regression statistics' in s
        assert '(ME) : 0.0000' in s
        assert '(RMSE) : 0.8944' in s
        assert '(MAE) : 0.8000' in s
        assert '(MPE) : -4.3333' in s
        assert '(MAPE) : 25.6667' in s

        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 2, 1, 4, 3]

        out = StringIO()
        with redirect_stdout(out):
            regressionSummary(y_true=y_true, y_pred=y_pred)
        s = out.getvalue()
        assert 'Regression statistics' in s
        assert '(ME) : 0.0000' in s
        assert '(RMSE) : 0.8944' in s
        assert '(MAE) : 0.8000' in s
        assert '(MPE)' not in s
        assert '(MAPE)' not in s

    def test_regressionSummary2(self) -> None:
        y_true = [[1], [2], [3], [4], [5]]
        y_pred = [[1], [3], [2], [5], [4]]

        out = StringIO()
        with redirect_stdout(out):
            regressionSummary(y_true=y_true, y_pred=y_pred)
        s = out.getvalue()
        assert 'Regression statistics' in s
        assert '(ME) : 0.0000' in s
        assert '(RMSE) : 0.8944' in s
        assert '(MAE) : 0.8000' in s
        assert '(MPE) : -4.3333' in s
        assert '(MAPE) : 25.6667' in s

    def test_classificationSummary(self) -> None:
        y_true = ['b' if y == 1 else 'a' for y in [1, 0, 0, 1, 1, 1]]
        y_pred = ['b' if y == 1 else 'a' for y in [1, 0, 1, 1, 0, 0]]

        out = StringIO()
        with redirect_stdout(out):
            classificationSummary(y_true=y_true, y_pred=y_pred, class_names=['a', 'b'])
        s = out.getvalue()

        assert 'Confusion Matrix' in s
        assert '       Prediction' in s
        assert 'a 1 1' in s
        assert 'b 2 2' in s
