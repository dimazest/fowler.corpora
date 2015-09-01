"""Implementation of common word sense disambiguation methods."""
import logging

import colored
import pandas as pd

from fowler.corpora.bnc.main import uri_to_corpus_reader
from fowler.corpora.bnc.readers import KS13

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceMixin
from fowler.corpora.models import read_space_from_file
from fowler.corpora.util import display

from .datasets import dataset_types
from .experiments import SimilarityExperiment


logger = logging.getLogger(__name__)


class WSDDispatcher(Dispatcher, SpaceMixin):
    """WSD task dispatcher."""

    global__composition_operator = (
        '',
        (
            'verb',
            'add',
            'mult',
            'kron',
            'relational',
            'copy-object',
            'copy-subject',
            'frobenious-add',
            'frobenious-mult',
            'frobenious-outer',
        ),
        'Composition operator.',
    )
    global__google_vectors = '', False, 'Get rid of certain words in the input data that are not in Google vectors.'
    global__verb_space = '', '', 'Separate transitive verb space.'
    global__dataset = ('', '', 'The kind of dataset.')

    @Resource
    def dataset(self):
        return uri_to_corpus_reader(self.kwargs['dataset'])

    @Resource
    def verb_space(self):
        if self.kwargs['verb_space']:
            return read_space_from_file(self.kwargs['verb_space'])

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


def gs11_similarity(args):
    (v, verb), (s, subject), (o, object_), (l, landmark), composition_operator = args

    if composition_operator == 'kron':
        subject_object = compose(subject, object_)

        sentence_verb = verb.multiply(subject_object)
        sentence_landmark = landmark.multiply(subject_object)
    elif composition_operator == 'mult':
        sentence_verb = verb.multiply(subject).multiply(object_)
        sentence_landmark = landmark.multiply(subject).multiply(object_)
    elif composition_operator == 'add':
        sentence_verb = verb + subject + object_
        sentence_landmark = landmark + subject + object_
    elif composition_operator == 'verb':
        sentence_verb = verb
        sentence_landmark = landmark

    return pairwise.cosine_similarity(sentence_verb, sentence_landmark)[0][0]


@command()
def gs11(
    pool,
    space,
    composition_operator,
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
                composition_operator,
            )
            for v, s, o, l in gs11_data[['verb', 'subject', 'object', 'landmark']].values
        ),
        similarity_function=gs11_similarity,
        input_column='input',
        composition_operator=composition_operator,
    )

    display(gs11_data.groupby('hilo').mean())


class Composer:
    # TODO: Rename, as *Vectorizer
    def __init__(self, space, operator):
        self.space = space
        self.operator = operator  # TODO: should be differenct classes!

    def compose(self, sentence):
        subject = self.space[sentence[0]]
        verb = self.space[sentence[1]]
        object_ = self.space[sentence[2]]

        def relational():
            return verb.multiply(kron(subject, object_, format='bsr'))

        def copy_object():
            return subject.T.multiply(verb.dot(object_.T)).T

        def copy_subject():
            return subject.dot(verb).multiply(object_)

        Sentence = {
            'verb': lambda: verb,
            'add': lambda: verb + subject + object_,
            'mult': lambda: verb.multiply(subject).multiply(object_),
            'kron': lambda: verb.multiply(kron(subject, object_, format='bsr')),
            'relational': relational,
            'copy-object': copy_object,
            'copy-subject': copy_subject,
            'frobenious-add': lambda: copy_object() + copy_subject(),
            'frobenious-mult': lambda: copy_object().multiply(copy_subject()),
            'frobenious-outer': lambda: kron(copy_object(), copy_subject()),

        }[self.operator]

        return Sentence()

    def info(self):
        return ' ({s.BOLD}{co}{s.RESET})'.format(
            s=colored.style,
            co=self.operator,
        )


@command()
def similarity(
        pool,
        dataset,
        no_p11n,
        composition_operator,
        space,
        verb_space,
        output=('o', 'sentence_similarity.h5', 'Result output file.'),
        key=('', 'dataset', 'The key of the result in the output file.')
):

    composer = Composer(space, composition_operator)
    experiment = SimilarityExperiment(show_progress_bar=not no_p11n, pool=pool)

    experiment.evaluate(
        dataset=dataset,
        composer=composer,
    ).to_hdf(output, key=key)


def gs12_similarity(args):
    (
        (as_, adj_subj),
        (s, subj),
        (v, verb),
        (l, landmark),
        (ao, adj_obj),
        (o, obj),
        composition_operator,
        np_composition,
    ) = args

    def compose(a, n):
        return {
            'add': lambda: a + n,
            'mult': lambda: a.multiply(n),
        }[np_composition]()

    return gs11_similarity(
        (
            (v, verb), (s, compose(adj_subj, subj)), (o, compose(adj_obj, obj)), (l, landmark), composition_operator,
        )
    )


def gs12(
    pool,
    space,
    composition_operator,
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
                composition_operator,
                np_composition,
            )
            for as_, s, v, l, ao, o in gs12_data[['adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj']].values
        ),
        similarity_function=gs12_similarity,
        input_column='annotator_score',
        composition_operator=composition_operator,
    )


@command()
def targets(
    dataset_class,
    dataset=('', 'downloads/compdistmeaning/emnlp2013_turk.txt', 'The KS2013 dataset.'),
    out=('o', 'targets.csv', 'KS targets'),
    tagset=('', '', 'Tagset'),
):
    """Extract target words from a dataset."""

    tokens = dataset_class.tokens(
        dataset_class.read(dataset),
        tagset,
    )

    tokens.drop_duplicates().to_csv(out, index=False)
