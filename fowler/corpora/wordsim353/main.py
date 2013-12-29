"""The WordSimilarity-353 Test."""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.metrics import pairwise
from scipy.stats import spearmanr


from fowler.corpora.util import display
from fowler.corpora.dispatcher import Dispatcher

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def evaluate(
    templates_env,
    matrix=('m', 'matrix.h5', 'The cooccurrence matrix.'),
    gold_standard=('g', 'wordsim353/combined.csv', 'Word similarities'),
):
    """Evaluate a distributional semantic vector space."""
    with pd.get_store(matrix, mode='r') as store:
        targets = store['targets']
        matrix = store['matrix'].reset_index()
        context_len = len(store['context'])

    result = (
        pd.read_csv(gold_standard)
        .merge(targets, left_on='Word 1', right_index=True)
        .merge(targets, left_on='Word 2', right_index=True, suffixes=('_word1', '_word2'))
    ).sort('Human (mean)', ascending=False)

    counts = matrix['count'].values
    ij = matrix[['id_target', 'id_context']].values.T

    space = csc_matrix((counts, ij), shape=(len(targets), context_len))

    def cosine_similarity(row):
        word1 = space[row['id_word1']]
        word2 = space[row['id_word2']]
        return pairwise.cosine_similarity(word1, word2)[0][0]

    def inner_product(row):
        word1 = space[row['id_word1']].toarray()
        word2 = space[row['id_word2']].toarray()

        return np.inner(word1, word2)[0][0]

    result['Cosine similarity'] = result.apply(cosine_similarity, axis=1)
    result['Inner (scalar) product similarity'] = result.apply(inner_product, axis=1)

    del result['id_word1']
    del result['id_word2']

    human = result['Human (mean)']
    cosine_similarity = result['Cosine similarity']

    display(
        templates_env.get_template('wordsim353_report.rst').render(
            cor_coof=(
                ('Cosine', spearmanr(human, cosine_similarity)),
                ('Inner product', spearmanr(human, result['Inner (scalar) product similarity'])),
            )
        )
    )

    display(result)
    _plot(human, cosine_similarity)


def _plot(x, y):
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

    plt.plot()
