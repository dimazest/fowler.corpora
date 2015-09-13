"""Implementation of common word sense disambiguation methods."""
import logging
from itertools import chain

import colored
import pandas as pd

from scipy import sparse

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
            'head',
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


def transitive_sentence(tagset):
    from .datasets import tag_mappings

    return (
        ('TOP', 'TOP'),
        (
            (
                (tag_mappings[tagset]['V'], 'ROOT'),
                (
                    ((tag_mappings[tagset]['N'], 'SBJ'), ()),
                    ((tag_mappings[tagset]['N'], 'OBJ'), ()),
                ),
            ),
        ),
    )


def graph_signature(dg, node=0):
    result = tuple()

    for dep_address in sorted(chain.from_iterable(dg.nodes[node]['deps'].values())):
        result += graph_signature(dg, dep_address),

    return (dg.nodes[node]['tag'], dg.nodes[node]['rel']), result


class CompositionalVectorizer:

    transitive_operators = {
        'kron', 'relational', 'copy-object', 'copy-subject',
        'frobenious-add', 'frobenious-mult', 'frobenious-outer',
    }

    def __init__(self, space, operator, tagset):
        self.space = space
        self.operator = operator  # TODO: should be differenct classes and register using entrypoints.
        self.tagset = tagset

    def vectorize(self, dependency_graph):
        nodes = dependency_graph.nodes

        if self.operator == 'head':
            assert len(nodes[0]['deps']) == 1

            head_address, = nodes[0]['deps']['ROOT']
            return self.node_to_vector(nodes[head_address])

        elif self.operator in ('add', 'mult'):
            tokens = tuple((node['lemma'], node['tag']) for node in nodes.values() if node['address'])
            return getattr(self.space, self.operator)(*tokens)

        elif self.operator in self.transitive_operators:
            assert graph_signature(dependency_graph) == transitive_sentence(self.tagset)

            subject = self.node_to_vector(nodes[1])
            verb = self.node_to_vector(nodes[2])
            object_ = self.node_to_vector(nodes[3])

            if self.operator == 'kron':
                verb_matrix = sparse.kron(verb, verb, format='csr')
                subject_object = sparse.kron(subject, object_, format='csr')

                return verb_matrix.multiply(subject_object)

            else:
                raise NotImplemented('Operator {} is not implemented'.format(self.operator))
        else:
            raise ValueError('Operator {} is not supported'.format(self.operator))

    def info(self):
        return '({s.BOLD}{operator}{s.RESET})'.format(
            s=colored.style,
            operator=self.operator,
        )

    def node_to_vector(self, node):
        return self.space[node['lemma'], node['tag']]


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

    if dataset.vectorizer == 'lexical':
        assert composition_operator == 'head'

    vectorizer = CompositionalVectorizer(
        space,
        composition_operator,
        tagset=dataset.tagset,
    )

    experiment = SimilarityExperiment(show_progress_bar=not no_p11n, pool=pool)

    experiment.evaluate(
        dataset=dataset,
        vectorizer=vectorizer,
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
