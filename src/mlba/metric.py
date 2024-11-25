'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import numpy.typing as npt
from typing import Any, cast
from sklearn.linear_model import LinearRegression
import math
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score


REGRESSION_YTYPES = Any
CLASSIFICATION_YTYPES = Any


def adjusted_r2_score(y_true: REGRESSION_YTYPES, y_pred: REGRESSION_YTYPES, model: LinearRegression) -> float:
    """ calculate adjusted R2
    Input:
        y_true: actual values
        y_pred: predicted values
        model: predictive model
    """
    n = len(y_pred)
    p = len(model.coef_)
    if p >= n - 1:
        return 0
    r2 = r2_score(y_true, y_pred)
    return cast(float, 1 - (1 - r2) * (n - 1) / (n - p - 1))


def AIC_score(*, y_true: REGRESSION_YTYPES | CLASSIFICATION_YTYPES, y_pred: REGRESSION_YTYPES | CLASSIFICATION_YTYPES,
              model: LinearRegression | None = None, df: int | None = None) -> float:
    """ calculate Akaike Information Criterion (AIC)
    Input:
        y_true: actual values
        y_pred: predicted values
        model (optional): predictive model
        df (optional): degrees of freedom of model

    One of model or df is requried
    """
    if df is None and model is None:
        raise ValueError('You need to provide either model or df')
    n = len(y_pred)
    p = degrees_of_freedom(model, df)
    if isinstance(list(y_true)[0], str):
        sse = sum(yt != yp for yt, yp in zip(y_true, y_pred))
    else:
        resid = np.array(y_true) - np.array(y_pred)
        sse = np.sum(resid ** 2)
    constant = n + n * np.log(2 * np.pi)
    return cast(float, n * math.log(sse / n) + constant + 2 * (p + 1))


def BIC_score(*, y_true: REGRESSION_YTYPES, y_pred: REGRESSION_YTYPES, model: LinearRegression | None = None,
              df: int | None = None) -> float:
    """ calculate Schwartz's Bayesian Information Criterion (AIC)
    Input:
        y_true: actual values
        y_pred: predicted values
        model: predictive model
        df (optional): degrees of freedom of model
    """
    aic = AIC_score(y_true=y_true, y_pred=y_pred, model=model, df=df)
    p = degrees_of_freedom(model, df)
    n = len(y_pred)
    return aic - 2 * (p + 1) + math.log(n) * (p + 1)


def degrees_of_freedom(model: LinearRegression | None, df: int | None) -> int:
    """ calculate degrees of freedom
    Input:
        model: predictive model
        df (optional): degrees of freedom of model
    """
    if df is not None:
        return df
    if model is None:
        raise ValueError('You need to provide either model or df')
    return len(model.coef_) + 1


def regressionSummary(*, y_true: REGRESSION_YTYPES, y_pred: REGRESSION_YTYPES) -> None:
    """ print regression performance metrics

    Input:
        y_true: actual values
        y_pred: predicted values
    """
    metrics = regressionMetrics(y_true=y_true, y_pred=y_pred)
    label = {
        'ME': 'Mean Error (ME)',
        'RMSE': 'Root Mean Squared Error (RMSE)',
        'MAE': 'Mean Absolute Error (MAE)',
        'MPE': 'Mean Percentage Error (MPE)',
        'MAPE': 'Mean Absolute Percentage Error (MAPE)',
    }
    print('\nRegression statistics\n')
    for metric, value in metrics.items():
        print(f'{label[metric]} : {value:.4f}')


def regressionMetrics(*, y_true: REGRESSION_YTYPES, y_pred: REGRESSION_YTYPES) -> dict[str, float]:
    """ calculate and return regression performance metrics

    Input:
        y_true: actual values
        y_pred: predicted values
    Output:
        dictionary of regression metrics
    """
    y_true = _toArray(y_true)
    y_pred = _toArray(y_pred)
    y_res = y_true - y_pred
    metrics = {
        'ME': sum(y_res) / len(y_res),
        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': sum(abs(y_res)) / len(y_res),
    }
    if all(yt != 0 for yt in y_true):
        metrics['MPE'] = 100 * sum(y_res / y_true) / len(y_res)
        metrics['MAPE'] = 100 * sum(abs(y_res / y_true) / len(y_res))
    return metrics


def _toArray(y: REGRESSION_YTYPES) -> npt.NDArray:
    y1 = np.asarray(y)
    if len(y1.shape) == 2 and y1.shape[1] == 1:
        y1 = y1.ravel()
    return y1


def classificationSummary(*, y_true: CLASSIFICATION_YTYPES, y_pred: CLASSIFICATION_YTYPES,
                          class_names: list[str] | None = None) -> None:
    """ Print a summary of classification performance

    Input:
        y_true: actual values
        y_pred: predicted values
        class_names (optional): list of class names
    """
    if class_names is None:
        class_names = sorted({*y_true, *y_pred})
    labels = class_names
    confusionMatrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)

    print(f'Confusion Matrix (Accuracy {accuracy:.4f})\n')

    # Pretty-print confusion matrix
    cm = confusionMatrix

    # Convert the confusion matrix and labels to strings
    cm = [[str(i) for i in row] for row in cm]
    labels = [str(i) for i in labels]

    # Determine the width for the first label column and the individual cells
    prediction = 'Prediction'
    actual = 'Actual'
    labelWidth = max(len(s) for s in labels)
    cmWidth = max(*(len(s) for row in cm for s in row), labelWidth) + 1
    labelWidth = max(labelWidth, len(actual))

    # Construct the format statements
    fmt1 = f'{{:>{labelWidth}}}'
    fmt2 = f'{{:>{cmWidth}}}' * len(labels)

    # And print the confusion matrix
    print(fmt1.format(' ') + ' ' + prediction)
    print(fmt1.format(actual), end='')
    print(fmt2.format(*labels))

    for cls, row in zip(labels, cm):
        print(fmt1.format(cls), end='')
        print(fmt2.format(*row))


def classificationMetrics(*, y_true: CLASSIFICATION_YTYPES, y_pred: CLASSIFICATION_YTYPES) -> dict[str, float]:
    """ Calculate and return classification metrics

    Input:
        y_true: actual values
        y_pred: predicted values
    """
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'accuracy': accuracy,
    }
