import csv

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
    """Write a space."""
    with pd.get_store(f_name, mode='w', complevel=9, complib='zlib') as store:
        store['context'] = context
        store['targets'] = targets
        store['matrix'] = matrix

    # TODO it would be nice to write metadata, e.g. the command the file
    # was generated, the date and so on.
