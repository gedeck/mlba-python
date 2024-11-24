"""
Utility functions for

Machine Learning for Business Analytics:
Concepts, Techniques, and Applications in Python

(c) 2019-2025 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""
import unittest

import pandas as pd

from mlba import liftChart, gainsChart, textDecisionTree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class TestGraphs(unittest.TestCase):
    def test_liftChart(self) -> None:
        data = pd.Series([7] * 10 + [2.5] * 10 + [0.5]
                         * 10 + [0.25] * 20 + [0.1] * 50)
        df = pd.DataFrame({'ranking': data, 'actual': range(len(data))})
        ax = liftChart(df, actual='actual', ranking='ranking')
        assert ax is not None

    def test_gainsChart(self) -> None:
        data = pd.Series([7] * 10 + [2.5] * 10 + [0.5]
                         * 10 + [0.25] * 20 + [0.1] * 50)
        df = pd.DataFrame({'ranking': data, 'actual': range(len(data))})
        ax = gainsChart(df, actual='actual', ranking='ranking')
        assert ax is not None

    def test_textDecisionTree(self) -> None:
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

        estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator.fit(X_train, y_train)

        representation = textDecisionTree(estimator)
        # print(representation)

        assert 'node=0 test node' in representation
        assert 'node=1 leaf node' in representation
        assert 'node=2 test node' in representation
        assert 'node=3 leaf node' in representation
        assert 'node=4 leaf node' in representation
