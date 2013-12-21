"""The WordSimilarity-353 Test."""

import pandas as pd
from scipy.sparse import csc_matrix

from fowler.corpora.main.options import Dispatcher


dispatcher = Dispatcher()
command = dispatcher.command


@command()
def evaluate(
    matrix=('m', 'matrix.h5', 'The cooccurrence matrix.'),
    gold_standard=('g', 'wordsim353/combined.csv', 'Word similarities'),
):
    """Evaluate a distributional semantic vector space."""
    with pd.get_store(matrix, mode='r') as store:
        targets = store['targets']
        matrix = store['matrix'].reset_index()

    gold_standard = (
        pd.read_csv(gold_standard)
        .merge(targets, left_on='Word 1', right_index=True)
        .merge(targets, left_on='Word 2', right_index=True)
    )

    counts = matrix['count'].values
    ij = matrix[['id_target', 'id_context']].values.T

    space = csc_matrix((counts, ij))

    import pdb; pdb.set_trace()

