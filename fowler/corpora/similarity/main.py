"""Similarity experimets."""
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.metrics import pairwise

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceMixin


class SimilarityDispatcher(Dispatcher, SpaceMixin):
    """Similarity task dispatcher."""

    @Resource
    def rg65_data(self):
        """The data set grouped by verb, subject, object and landmark.

        The mean of the input values per group is calculated.

        """
        data = pd.read_csv(
            self.kwargs['rg65_data'],
            sep='\t',
            header=None,
            names=('Word 1', 'Word 2', 'Human'),
        )

        if self.limit:
            data = data.head(self.limit)

        return data

    @Resource
    def wordsim353_data(self):
        data = pd.read_csv(self.kwargs['wordsim353_data'])

        # for column in 'Word 1', 'Word 2':
            # data[column] = data[column].str.lower()

        # data.replace('troops', 'troop', inplace=True)

        return data

    @property
    def similarity_experiment(self):
        def experiment(similarity_input, data, input_column, *, space=self.space):
            """Calculate similarities between items.

            :param similarity_input: accessor to the vectors of the passed data.
            :type similarity_input: data -> iter(v1, v2)

            """
            # TODO inner product.
            similarities = self.pool.imap(
                cosine_similarity,
                similarity_input(data, space),
            )

            data['Cosine similarity'] = list(similarities)

            print(
                'Spearman correlation: rho={0:.3f}, p={1:.3f}'
                .format(*stats.spearmanr(data[[input_column, 'Cosine similarity']]))
            )

            sns.jointplot(
                data[input_column],
                data['Cosine similarity'],
                kind='reg',
                stat_func=stats.spearmanr,
                xlim=(1, 7),
                ylim=(0, 1),
            )

        return experiment


dispatcher = SimilarityDispatcher()
command = dispatcher.command


def cosine_similarity(words):
    word1, word2 = words
    return pairwise.cosine_similarity(word1, word2)[0][0]


@command()
def rg65(
    similarity_experiment,
    rg65_data=('', 'downloads/RubensteinGoodenough/EN-RG-65.txt', 'The rg65 dataset.'),

):
    """The Rubenstein and Goodenough noun similarity experiment [1]

    65 noun pairs with human similarity ratings.

    [1] Herbert Rubenstein and John B. Goodenough. 1965. Contextual correlates
        of synonymy. Commun. ACM 8, 10 (October 1965), 627-633.
        DOI=10.1145/365628.365657 http://doi.acm.org/10.1145/365628.365657

    [2] http://www.cs.cmu.edu/~mfaruqui/suite.html

    """
    similarity_experiment(
        lambda data, space: ((space[w1], space[w2]) for w1, w2 in data[['Word 1', 'Word 2']].values),
        rg65_data,
        input_column='Human'
    )


@command()
def wordsim353(
    similarity_experiment,
    wordsim353_data=('', 'downloads/wordsim353/combined.csv', 'The wordsim 353 dataset.'),
):
    """The worsim353 similarity experiment."""
    similarity_experiment(
        lambda data, space: ((space[w1], space[w2]) for w1, w2 in data[['Word 1', 'Word 2']].values),
        wordsim353_data,
        input_column='Human (mean)'
    )

