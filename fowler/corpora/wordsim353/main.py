"""The WordSimilarity-353 Test."""

import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.metrics import pairwise
from scipy.stats import spearmanr
from numpy import corrcoef

from IPython.display import display

from fowler.corpora.dispatcher import Dispatcher

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

    result = (
        pd.read_csv(gold_standard)
        .merge(targets, left_on='Word 1', right_index=True)
        .merge(targets, left_on='Word 2', right_index=True, suffixes=('_word1', '_word2'))
    ).sort('Human (mean)', ascending=False)

    counts = matrix['count'].values
    ij = matrix[['id_target', 'id_context']].values.T

    def cosine_similarity(row):
        word1 = space[row['id_word1']]
        word2 = space[row['id_word2']]
        return pairwise.cosine_similarity(word1, word2)[0][0]

    space = csc_matrix((counts, ij))

    result['cosine_similarity'] = result.apply(cosine_similarity, axis=1)
    del result['id_word1']
    del result['id_word2']

    human = result['Human (mean)']
    cosine_similarity = result['cosine_similarity']

    print(
        'Spearman rho={spearman[0]:0.3}, p-value={spearman[1]:0.3}\n'
        '\n'
        'Correlation coefficients:\n'
        '{corrcoef}'
        ''.format(
            spearman=spearmanr(human, cosine_similarity),
            corrcoef=corrcoef(human, cosine_similarity),
        )
    )

    display(result)
    _plot(human, cosine_similarity)


def _plot(x, y):
    import numpy as np
    import matplotlib.pyplot as plt

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(1, figsize=(10, 10))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    s = 0
    e = 10

    axScatter.set_xlim((s, e))
    axScatter.set_ylim((s, e / 10))

    bins_x = np.arange(s, e + binwidth, binwidth)
    axHistx.hist(x, bins=bins_x)

    bins_y = np.arange(s, e + binwidth / 10, binwidth / 10)
    axHisty.hist(y, bins=bins_y, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    display(plt)
