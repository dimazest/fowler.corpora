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

    @Resource
    def ks13_data(self):
        """
        Paraphrasing dataset provided by [1] and re-scored by [2].

        [1] Mitchell, J., & Lapata, M. (2010). Composition in distributional
            models of semantics. Cognitive science, 34(8), 1388-1429.

        [2] Kartsaklis, D., & Sadrzadeh, M. Prior Disambiguation of Word Tensors
            for Constructing Sentence Vectors.

        """
        index_cols = 'subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2'

        data = pd.read_csv(
            self.kwargs['ks13_data'],
            sep=' ',
            usecols=index_cols + ('score', ),
        )

        grouped = data.groupby(index_cols, as_index=False).mean()

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
    similarity_experiment(
        space,
        pool,
        gs11_data,
        verb_columns=('verb', 'landmark'),
        similarity_input=lambda verb_vectors, S: (
            (
                (v, verb_vectors[v]),
                (s, space[S(s)]),
                (o, space[S(o)]),
                (l, verb_vectors[l]),
            )
            for v, s, o, l in gs11_data[['verb', 'subject', 'object', 'landmark']].values
        ),
        similarity_function=gs11_similarity,
        input_column='input'
    )

    display(gs11_data.groupby('hilo').mean())


@command()
def paraphrasing(
    pool,
    space,
    ks13_data=('', 'downloads/compdistmeaning/emnlp2013_turk.txt', 'The KS2013 dataset.'),
):
    similarity_experiment(
        space,
        pool,
        ks13_data,
        verb_columns=('verb1', 'verb2'),
        similarity_input=lambda verb_vectors, S: (
            (
                (s1, space[S(s1)]),
                (v1, verb_vectors[v1]),
                (o1, space[S(o1)]),
                (s2, space[S(s2)]),
                (v2, verb_vectors[v2]),
                (o2, space[S(o2)]),
            )
            for s1, v1, o1, s2, v2, o2 in ks13_data[['subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2']].values
        ),
        similarity_function=paraphrasing_similarity,
        input_column='score'
    )


def paraphrasing_similarity(args):
    (s1, subject1), (v1, verb1), (o1, object1), (s2, subject2), (v2, verb2), (o2, object2) = args

    def Sentence(subject, verb, object_):
        return verb.multiply(compose(subject, object_))

    s1 = Sentence(subject1, verb1, object1)
    s2 = Sentence(subject2, verb2, object2)

    return pairwise.cosine_similarity(s1, s2)[0][0]


def similarity_experiment(space, pool, data, verb_columns, similarity_input, similarity_function, input_column):
    def T(w, tag):
        if space.row_labels.index.nlevels == 2:
            return w, tag

        return w

    V = lambda v: T(v, 'VERB')
    S = lambda s: T(s, 'SUBST')

    verbs = pd.concat([data[vc] for vc in verb_columns]).unique()
    verb_vectors = Bar(
        'Verb vectors',
        max=len(verbs),
        suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
    ).iter(
        zip(verbs, (pool.starmap(compose, ((space[V(v)], space[V(v)]) for v in verbs))))
    )
    verb_vectors = dict(verb_vectors)

    similarities = pool.imap(
        similarity_function,
        similarity_input(verb_vectors, S)
    )

    data['Cosine similarity'] = list(
        Bar(
            'Cosine similarity',
            max=(len(data)),
            suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
        ).iter(similarities)
    )

    sns.jointplot(
        data[input_column],
        data['Cosine similarity'],
        kind='reg',
        stat_func=stats.spearmanr,
        xlim=(1, 7),
        ylim=(0, 1),
    )

    print(
        'Spearman correlation: rho={0:.2f}, p={1:.2f}'
        .format(*stats.spearmanr(data[[input_column, 'Cosine similarity']]))
    )

    return data
