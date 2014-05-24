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

    global__compositon_operator = '', 'kron', 'Composition operator [kron|sum|mult|verb].'
    global__google_vectors = '', False, 'Get rid of certain words in the input data that are not in Google vectors.'

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

        if self.google_vectors:
            grouped = grouped[grouped['landmark'] != 'mope']
            grouped = grouped[grouped['object'] != 'behaviour']
            grouped = grouped[grouped['object'] != 'favour']
            grouped = grouped[grouped['object'] != 'offence']
            grouped = grouped[grouped['object'] != 'paper']

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

        if self.google_vectors:
            grouped = grouped[grouped['verb2'] != 'emphasise']
            grouped = grouped[grouped['subject1'] != 'programme']
            grouped = grouped[grouped['subject2'] != 'programme']

        if self.limit:
            grouped = grouped.head(self.limit)

        return grouped

    @Resource
    def gs12_data(self):
        """The data set grouped by 'adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj'.

        The mean of the input values per group is calculated.

        """
        index_cols = 'adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj'

        data = pd.read_csv(
            self.kwargs['gs12_data'],
            sep=' ',
            usecols=index_cols + ('annotator_score', ),
        )
        grouped = data.groupby(index_cols, as_index=False).mean()

        grouped['obj'][grouped['obj'] == 'papers'] = 'paper'

        if self.google_vectors:
            grouped = grouped[grouped['obj'] != 'offence']
            grouped = grouped[grouped['obj'] != 'favour']

        if self.limit:
            grouped = grouped.head(self.limit)

        return grouped


dispatcher = WSDDispatcher()
command = dispatcher.command


def compose(a, b):
    return sparse.kron(a, b, format='bsr')


def gs11_similarity(args):
    (v, verb), (s, subject), (o, object_), (l, landmark), compositon_operator = args

    if compositon_operator == 'kron':
        subject_object = compose(subject, object_)

        sentence_verb = verb.multiply(subject_object)
        sentence_landmark = landmark.multiply(subject_object)
    elif compositon_operator == 'mult':
        sentence_verb = verb.multiply(subject).multiply(object_)
        sentence_landmark = landmark.multiply(subject).multiply(object_)
    elif compositon_operator == 'add':
        sentence_verb = verb + subject + object_
        sentence_landmark = landmark + subject + object_
    elif compositon_operator == 'verb':
        sentence_verb = verb
        sentence_landmark = landmark

    return pairwise.cosine_similarity(sentence_verb, sentence_landmark)[0][0]


@command()
def gs11(
    pool,
    space,
    compositon_operator,
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
        similarity_input=lambda verb_vectors, t: (
            (
                (v, verb_vectors[v]),
                (s, space[t.S(s)]),
                (o, space[t.S(o)]),
                (l, verb_vectors[l]),
                compositon_operator,
            )
            for v, s, o, l in gs11_data[['verb', 'subject', 'object', 'landmark']].values
        ),
        similarity_function=gs11_similarity,
        input_column='input',
        compositon_operator=compositon_operator,
    )

    display(gs11_data.groupby('hilo').mean())


@command()
def paraphrasing(
    pool,
    space,
    compositon_operator,
    ks13_data=('', 'downloads/compdistmeaning/emnlp2013_turk.txt', 'The KS2013 dataset.'),
):
    similarity_experiment(
        space,
        pool,
        ks13_data,
        verb_columns=('verb1', 'verb2'),
        similarity_input=lambda verb_vectors, t: (
            (
                (s1, space[t.S(s1)]),
                (v1, verb_vectors[v1]),
                (o1, space[t.S(o1)]),
                (s2, space[t.S(s2)]),
                (v2, verb_vectors[v2]),
                (o2, space[t.S(o2)]),
                compositon_operator,
            )
            for s1, v1, o1, s2, v2, o2 in ks13_data[['subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2']].values
        ),
        similarity_function=paraphrasing_similarity,
        input_column='score',
        compositon_operator=compositon_operator,
    )


def gs12_similarity(args):
    (
        (as_, adj_subj),
        (s, subj),
        (v, verb),
        (l, landmark),
        (ao, adj_obj),
        (o, obj),
        compositon_operator,
        np_composition,
    ) = args

    def compose(a, n):
        return {
            'add': lambda: a + n,
            'mult': lambda: a.multiply(n),
        }[np_composition]()

    return gs11_similarity(
        (
            (v, verb), (s, compose(adj_subj, subj)), (o, compose(adj_obj, obj)), (l, landmark), compositon_operator,
        )
    )


@command()
def gs12(
    pool,
    space,
    compositon_operator,
    np_composition=('', 'mult', 'Operation used to compose adjective with noun. [add|mult]'),
    gs12_data=('', 'downloads/compdistmeaning/GS2012data.txt', 'The GS2012 dataset.'),
):
    similarity_experiment(
        space,
        pool,
        gs12_data,
        verb_columns=('verb', 'landmark'),
        similarity_input=lambda verb_vectors, t: (
            (
                (as_, space[t.A(as_)]),
                (s, space[t.S(s)]),
                (v, verb_vectors[v]),
                (l, verb_vectors[l]),
                (ao, space[t.A(ao)]),
                (o, space[t.S(o)]),
                compositon_operator,
                np_composition,
            )
            for as_, s, v, l, ao, o in gs12_data[['adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj']].values
        ),
        similarity_function=gs12_similarity,
        input_column='annotator_score',
        compositon_operator=compositon_operator,
    )


def paraphrasing_similarity(args):
    (s1, subject1), (v1, verb1), (o1, object1), (s2, subject2), (v2, verb2), (o2, object2), compositon_operator = args

    Sentence = {
        'kron': lambda subject, verb, object_: verb.multiply(compose(subject, object_)),
        'add': lambda subject, verb, object_: verb + subject + object_,
        'mult': lambda subject, verb, object_: verb.multiply(subject).multiply(object_),
        'verb': lambda subject, verb, object_: verb,
    }[compositon_operator]

    s1 = Sentence(subject1, verb1, object1)
    s2 = Sentence(subject2, verb2, object2)

    return pairwise.cosine_similarity(s1, s2)[0][0]


def similarity_experiment(space, pool, data, verb_columns, similarity_input, similarity_function, input_column, compositon_operator):

    tagger = Tagger(space)
    V = tagger.V

    verbs = pd.concat([data[vc] for vc in verb_columns]).unique()

    if compositon_operator == 'kron':
        verb_vectors = Bar(
            'Verb vectors',
            max=len(verbs),
            suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
        ).iter(
            zip(verbs, (pool.starmap(compose, ((space[V(v)], space[V(v)]) for v in verbs))))
        )
        verb_vectors = dict(verb_vectors)
    else:
        verb_vectors = {v: space[V(v)] for v in verbs}

    similarities = pool.imap(
        similarity_function,
        similarity_input(verb_vectors, tagger)
    )

    data['Cosine similarity'] = list(
        Bar(
            'Cosine similarity',
            max=(len(data)),
            suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
        ).iter(similarities)
    )

    print(
        'Spearman correlation: rho={0:.3f}, p={1:.2f}'
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

    return data


class Tagger:
    # TODO: it would be cool to move it somewhere

    def __init__(self, space):
        self.with_tags = space.row_labels.index.nlevels == 2

    def tag(self, w, tag):

        if (w, tag) == ('parish', 'ADJ'):
            tag = 'SUBST'
        if (w, tag) == ('raffle', 'ADJ'):
            tag = 'SUBST'

        if self.with_tags:
            return w, tag

        return w

    def V(self, v):
        """Verb."""
        return self.tag(v, 'VERB')

    def S(self, v):
        """Substantive (noun)."""
        return self.tag(v, 'SUBST')

    def A(self, v):
        """Adjective."""
        return self.tag(v, 'ADJ')
