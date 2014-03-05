"""Implementation of common word sense disambiguation methods."""
import logging

import pandas as pd
import seaborn as sns

from scipy import sparse, stats
from sklearn.metrics import pairwise

from zope.cachedescriptors import method
from progress.bar import Bar

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceMixin
from fowler.corpora.util import display


logger = logging.getLogger(__name__)


class WSDDispatcher(Dispatcher, SpaceMixin):
    """WSD task dispatcher."""

    @Resource
    def gs11_data(self):
        """The data set grouped by verb, subject, object and landmark.

        The mean of the input values per group is calculated.

        """
        data = pd.read_csv(self.kwargs['gs11_data'], sep=' ')
        grouped = data.groupby(
            ('verb', 'subject', 'object', 'landmark', 'hilo'),
            as_index=False,
        ).mean()

        if self.gs11_limit:
            grouped = grouped.head(self.gs11_limit)

        return grouped


dispatcher = WSDDispatcher()
command = dispatcher.command


class Composer():
    cache = {}

    def __init__(self, vector, other_vector):
        self.vector = vector
        self.other_vector = other_vector

    @method.cachedIn('cache')
    def compose(self, word, other_word):
        logger.debug('Verb matrix for %s and %s', word, other_word)
        return sparse.kron(self.vector, self.other_vector)


def gs11_compose(args):
    verb, subject, object_, landmark = args

    def compute_tensor_dot(this, other=None):
        word, vector = this

        if other is None:
            other = this
        other_word, other_vector = other

        return Composer(vector, other_vector).compose(word, other_word)

    subject_object = compute_tensor_dot(subject, object_)

    sentence_verb = compute_tensor_dot(verb).multiply(subject_object)
    sentence_landmark = compute_tensor_dot(landmark).multiply(subject_object)

    return pairwise.cosine_similarity(sentence_verb, sentence_landmark)[0][0]


@command()
def gs11(
    pool,
    space,
    gs11_data=('', 'downloads/compdistmeaning/GS2011data.txt', 'The GS2011 dataset.'),
    gs11_limit=('', 0, 'Limit number of items in the data set.'),
):
    """Categorical compositional distributional model for transitive verb disambiguation.

    Implements method described in [1]. The data is available at [2].

    [1] Grefenstette, Edward, and Mehrnoosh Sadrzadeh. "Experimental support
    for a categorical compositional distributional model of meaning."
    Proceedings of the Conference on Empirical Methods in Natural Language
    Processing. Association for Computational Linguistics, 2011.

    [2] http://www.cs.ox.ac.uk/activities/compdistmeaning/GS2011data.txt

    """
    result = pool.imap(
        gs11_compose,
        (
            ((v, space[v]), (s, space[s]), (o, space[o]), (l, space[l]))
            for v, s, o, l in gs11_data[['verb', 'subject', 'object', 'landmark']].values
        ),
    )

    bar = Bar(
        'Cosine similarity',
        max=(len(gs11_data)),
        suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
    )

    gs11_data['Cosine similarity'] = list(bar.iter(result))

    display(gs11_data.groupby('hilo').mean())
    sns.regplot(
        gs11_data['input'],
        gs11_data['Cosine similarity'],
        corr_func=stats.spearmanr,
    )
