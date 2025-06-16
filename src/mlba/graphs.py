"""
Utility functions for

Machine Learning for Business Analytics:
Concepts, Techniques, and Applications in Python

(c) 2019-2025 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""
from collections.abc import Iterable
import io
from os import PathLike
from typing import Literal, NamedTuple
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.tree import DecisionTreeClassifier, export_graphviz
try:
    from IPython.display import Image
except ImportError:
    Image = None
try:
    import graphviz
except ImportError:
    graphviz = None


def liftChart(data: pd.DataFrame, *, ranking: str | None = None, actual: str | None = None,
              title: str = 'Decile-wise lift chart', labelBars: bool = True,
              ax: Axes | None = None, figsize: Iterable[float] | None = None) -> Axes:
    """ Create a decile lift chart using ranking and predicted values

    Input:
        data: DataFrame with ranking and actual values
        ranking: column name for ranking (predicted values or probability)
        actual: column name for actual values
        title (optional): set to None to suppress title
        labelBars (optional): set to False to avoid mean response labels on bar chart
        ax (optional): axis for matplotlib graph
        figsize (optional): size of matplotlib graph
    """
    if ranking is None or actual is None:
        raise ValueError('Column names for ranking and actual must be provided')

    data = data.sort_values(by=[ranking], ascending=False)
    ranked_actual = data[actual]

    # group the sorted actual values into 10 roughly equal groups and calculate the mean
    groups = [int(10 * i / len(ranked_actual))
              for i in range(len(ranked_actual))]
    meanPercentile = ranked_actual.groupby(groups).mean()
    # divide by the mean prediction to get the mean response
    meanResponse = meanPercentile / ranked_actual.mean()
    meanResponse.index = (meanResponse.index + 1) * 10

    ax = meanResponse.plot.bar(color='C0', ax=ax, figsize=figsize)
    assert ax is not None
    ax.set_ylim(0, 1.12 * meanResponse.max() if labelBars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Decile mean / global mean')
    if title:
        ax.set_title(title)

    if labelBars:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x(), p.get_height() + 0.1))
    return ax


def gainsChart(data: pd.DataFrame, *, ranking: str | None = None, actual: str | None = None,
               event_level: int | str = 1, type: Literal['classification', 'regression'] = 'classification',  # noqa: A002
               color: str = 'C0', title: str = 'Cumulative gains chart', label: str | None = None,
               show_counts: bool = False, ax: Axes | None = None, figsize: Iterable[float] | None = None) -> Axes:
    """ Create a gains chart using ranking and predicted values

    Input:
        data: DataFrame with ranking and actual values
        ranking: column name for ranking (predicted values or probability)
        actual: column name for actual values
        event_level: outcome of interest for the actual values (default 1)
        type: classification (default) or regression
        color (optional): color of graph
        title (optional): set to None to suppress title
        show_counts (optional): show counts of cumulative gains (classification only)
        ax (optional): axis for matplotlib graph
        figsize (optional): size of matplotlib graph
    """
    if ranking is None or actual is None:
        raise ValueError('Column names for ranking and actual must be provided')

    gainsChartData = _getGainsChartData(data, ranking=ranking, actual=actual,
                                        event_level=event_level, type=type,
                                        show_counts=show_counts)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    assert ax is not None
    if type == 'classification':
        ax.plot([0, gainsChartData.optimal_gains * gainsChartData.nTotal, gainsChartData.nTotal],
                [0, gainsChartData.nActual, gainsChartData.nActual],
                color='lightgrey')
    ax = gainsChartData.gains_df.plot(x='records', y='cumGains', color=color, label=label, legend=False,
                                      ax=ax)

    # Add line for random gain
    ax.plot([0, gainsChartData.nTotal],
            [0, gainsChartData.nActual], linestyle='--', color='k')
    ax.set_title(title)
    ax.set_xlabel(gainsChartData.xlabel)
    ax.set_ylabel(gainsChartData.ylabel)
    return ax


def _getGainsChartData(data: pd.DataFrame, *, ranking: str | None = None, actual: str | None = None,
                      event_level: int | str = 1, type: Literal['classification', 'regression'] = 'classification',  # noqa: A002
                      show_counts: bool = False,
                      ) -> 'GainsChartData':
    data = data.sort_values(by=[ranking], ascending=False, ignore_index=True)
    if type == 'classification':
        gains = pd.Series([1 if g == event_level else 0 for g in data[actual]])
        nActual = gains.sum()  # number of desired records
        nTotal = len(gains)  # number of records
        optimal_gains = nActual / nTotal
    else:
        gains = data[actual]
        # nActual = len(gains)
        nActual = gains.sum()
        nTotal = len(gains)  # number of records
        optimal_gains = 1.0

    # get cumulative sum of gains and convert to percentage
    # Note the additional 0 at the front
    cumGains = pd.concat([pd.Series([0]), gains.cumsum()])
    gains_df = pd.DataFrame(
        {'records': list(range(nTotal + 1)), 'cumGains': cumGains})

    xlabel = '# records'
    if type == 'classification' and not show_counts:
        gains_df['records'] = 100 * gains_df['records'] / nTotal
        gains_df['cumGains'] = 100 * gains_df['cumGains'] / nActual
        nTotal = 100
        nActual = 100
        xlabel = 'Percent of cases'

    if show_counts:
        ylabel = '# cumulative gains'
    else:
        ylabel = 'Percent of positive cases'

    return GainsChartData(
        gains_df=gains_df,
        xlabel=xlabel,
        ylabel=ylabel,
        nTotal=nTotal,
        nActual=nActual,
        optimal_gains=optimal_gains,
    )


class GainsChartData(NamedTuple):
    gains_df: pd.DataFrame
    xlabel: str
    ylabel: str
    nTotal: int
    nActual: int
    optimal_gains: float


def plotDecisionTree(decisionTree: DecisionTreeClassifier, *, feature_names: list[str] | None = None,
                     class_names: list[str] | None = None, impurity: bool = False, label: str = 'root',
                     max_depth: int | None = None, rotate: bool = False,
                     pdfFile: PathLike | str | None = None) -> Image:
    """ Create a plot of the scikit-learn decision tree and show in the Jupyter notebook
    Input:
        decisionTree: scikit-learn decision tree
        feature_names (optional): variable names
        class_names (optional): class names, only relevant for classification trees
        impurity (optional): show node impurity
        label (optional): only show labels at the root
        max_depth (optional): limit
        rotate (optional): rotate the layout of the graph
        pdfFile (optional): provide pathname to create a PDF file of the graph
    """
    if graphviz is None:
        return 'You need to install graphviz and the graphviz package to visualize decision trees'
    if Image is None:
        return 'You need to install ipython to visualize decision trees'
    if class_names is not None:
        class_names = [str(s) for s in class_names]  # convert to strings
    dot_data = io.StringIO()
    export_graphviz(decisionTree, feature_names=feature_names, class_names=class_names, impurity=impurity,
                    label=label, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    max_depth=max_depth, rotate=rotate)
    graph = graphviz.Source(dot_data.getvalue())
    with TemporaryDirectory() as tempdir:
        if pdfFile is not None:
            graph.render('dot', directory=tempdir, format='pdf', outfile=pdfFile)
        return Image(graph.render('dot', directory=tempdir, format='png'))


def textDecisionTree(decisionTree: DecisionTreeClassifier, indent: str = '  ', *, as_ratio: bool = True) -> str:
    """ Create a text representation of the scikit-learn decision tree
    Input:
        decisionTree: scikit-learn decision tree
        as_ratio: show the composition of the leaf nodes as ratio (default) instead of counts
        indent: indentation (default two spaces)
    """
    # Taken from scikit-learn documentation
    n_nodes = decisionTree.tree_.node_count
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    threshold = decisionTree.tree_.threshold
    node_value = decisionTree.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))  # noqa: FURB113
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    rep = []
    for i in range(n_nodes):
        common = f'{node_depth[i] * indent}node={i}'
        if is_leaves[i]:
            value = node_value[i]
            if as_ratio:
                value = [[round(vi / sum(v), 3) for vi in v] for v in value]
            rep.append(f'{common} leaf node: {value}')
        else:
            rule = f'{children_left[i]} if {feature[i]} <= {threshold[i]} else to node {children_right[i]}'
            rep.append(f'{common} test node: go to node {rule}')
    return '\n'.join(rep)
