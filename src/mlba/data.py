"""
Utility functions for

Machine Learning for Business Analytics:
Concepts, Techniques, and Applications in Python

(c) 2019-2025 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""
from pathlib import Path
from typing import Any, cast

import pandas as pd


DATA_DIR = Path(__file__).parent / 'csvFiles'


def load_data(name: str, **kwargs: Any) -> pd.DataFrame | pd.Series:
    """ Returns the data either as a Pandas data frame or series """
    data_file = get_data_file(name)
    if not data_file.exists():
        raise ValueError(f'Data file {name} not found')
    data = cast(pd.DataFrame, pd.read_csv(data_file, **kwargs))
    if data.shape[1] == 1:
        return data[data.columns[0]]
    return data


def get_data_file(name: str) -> Path:
    if name.endswith('.zip'):
        return DATA_DIR / name
    name = name.removesuffix('.gz')
    name = name.removesuffix('.csv')
    return DATA_DIR / f'{name}.csv.gz'
