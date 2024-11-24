"""
Utility functions for

Machine Learning for Business Analytics:
Concepts, Techniques, and Applications in Python

(c) 2019-2025 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""
from typing import Any
import urllib
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def printTermDocumentMatrix(count_vect: CountVectorizer, counts: Any) -> None:
    """ Print term-document matrix created by the CountVectorizer
    Input:
        count_vect: scikit-learn Count vectorizer
        counts: term-document matrix returned by transform method of counter vectorizer
    """
    shape = counts.shape
    columns = [f'S{i}' for i in range(1, shape[0] + 1)]
    print(pd.DataFrame(data=counts.toarray().transpose(),
                       index=count_vect.get_feature_names_out(), columns=columns))


def downloadGloveModel() -> None:
    """ Download the GloVe model """
    if not os.path.exists('glove.6B.zip'):
        urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip',  # type: ignore[attr-defined]
                                   'glove.6B.zip')
