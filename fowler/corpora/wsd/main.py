"""Implementation of common word sense disambiguation methods."""
import logging

import pandas as pd
import seaborn as sns

from scipy import sparse, stats
from sklearn.metrics import pairwise

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

        if self.limit:
            grouped = grouped.head(self.limit)

        return grouped


dispatcher = WSDDispatcher()
command = dispatcher.command


def compose(a, b):
    return sparse.kron(a, b, format='bsr')


def gs11_similarity(args):
    (v, verb), (s, subject), (o, object_), (l, landmark) = args

    subject_object = compose(subject, object_)

    sentence_verb = verb.multiply(subject_object)
    sentence_landmark = landmark.multiply(subject_object)

    return pairwise.cosine_similarity(sentence_verb, sentence_landmark)[0][0]


@command()
def gs11(
    pool,
    space,
    gs11_data=('', 'downloads/compdistmeaning/GS2011data.txt', 'The GS2011 dataset.'),
):
    """Categorical compositional distributional model for transitive verb disambiguation.

    Implements method described in [1]. The data is available at [2].

    [1] Grefenstette, Edward, and Mehrnoosh Sadrzadeh. "Experimental support
    for a categorical compositional distributional model of meaning."
    Proceedings of the Conference on Empirical Methods in Natural Language
    Processing. Association for Computational Linguistics, 2011.

    [2] http://www.cs.ox.ac.uk/activities/compdistmeaning/GS2011data.txt

    """

    def T(w, tag):
        if space.row_labels.index.nlevels == 2:
            return w, tag

        return w

    V = lambda v: T(v, 'VERB')
    S = lambda s: T(s, 'SUBST')

    verbs = pd.concat([gs11_data['verb'], gs11_data['landmark']]).unique()
    verb_vectors = Bar(
        'Verb vectors',
        max=len(verbs),
        suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
    ).iter(
        zip(verbs, (pool.starmap(compose, ((space[V(v)], space[V(v)]) for v in verbs))))
    )
    verb_vectors = dict(verb_vectors)

    similarities = pool.imap(
        gs11_similarity,
        (
            (
                (v, verb_vectors[v]),
                (s, space[S(s)]),
                (o, space[S(o)]),
                (l, verb_vectors[l]),
            )
            for v, s, o, l in gs11_data[['verb', 'subject', 'object', 'landmark']].values
        ),
    )

    gs11_data['Cosine similarity'] = list(
        Bar(
            'Cosine similarity',
            max=(len(gs11_data)),
            suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
        ).iter(similarities)
    )

    display(gs11_data.groupby('hilo').mean())
    sns.jointplot(
        gs11_data['input'],
        gs11_data['Cosine similarity'],
        kind='reg',
        stat_func=stats.spearmanr,
        xlim=(1, 7),
        ylim=(0, 1),
    )

    print(
        'Spearman correlation: rho={0:.2f}, p={1:.2f}'
        .format(*stats.spearmanr(gs11_data[['input', 'Cosine similarity']]))
    )
