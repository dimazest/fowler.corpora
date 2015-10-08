import csv

import numpy as np
import pandas as pd


def read_tokens(f_name):
    """Read the series of tokens in a one column csv file."""
    return pd.read_csv(
        f_name,
        names=('ngram', ),
        index_col='ngram',
        encoding='utf8',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )


def write_space(f_name, context, targets, matrix):
    """Write a vector space without creating it.

    :param pandas.DataFrame context: the column labels
    :param pandas.DataFrame targets: the row lables

    ``row_labels`` and ``column_labels`` contain two columns: ``ngram`` and
    ``id``. ``ngram`` is frame's index.

    :param pandas.DataFrame matrix: the co-occurrence counts. The frame consists
        of three columns: ``count``, ``id_target`` and ``id_context``.

        ``id_target`` and ``id_context`` is frame's index.

    """

    assert np.isfinite(matrix['count']).all()

    assert context.reset_index().notnull().all().all(), 'There are NULL row labels.'
    assert targets.reset_index().notnull().all().all(), 'There are NULL column labels.'

    with pd.get_store(
        f_name,
        mode='w',
        complevel=9,
        complib='blosc',
    ) as store:
        # Reset index for context and tartgets, so the string "nan" is not
        # converted to NULL! Once https://github.com/pydata/pandas/issues/9604
        # is fixed, there is no need to reset the index.
        if context is not None:
            store['context'] = context.reset_index()

        store['targets'] = targets.reset_index()
        store['matrix'] = matrix

    # TODO it would be nice to write metadata, e.g. the command the file
    # was generated, the date and so on.
