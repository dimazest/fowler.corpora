"""Implementation of common word sense disambiguation methods."""
import logging
from itertools import chain

import colored
import pandas as pd
import numpy as np

from numpy import matlib
from scipy import sparse
from scipy.stats import entropy

from fowler.corpora.bnc.main import uri_to_corpus_reader

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceMixin
from fowler.corpora.models import read_space_from_file

from .experiments import SimilarityExperiment, EntailmentDirectionExperiment


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


dispatcher = WSDDispatcher()
command = dispatcher.command


def transitive_sentence(tagset):
    from .datasets import tag_mappings

    return (
        ('TOP', None),
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

    def __init__(self, space, operator, tagset, verb_space=None, sloppy=True, high_dim_kron=False, ignore_subject=False):
        self.space = space
        self.operator = operator  # TODO: should be differenct classes and register using entrypoints.
        self.tagset = tagset

        self.verb_space = verb_space
        self.sloppy = sloppy
        self.high_dim_kron = high_dim_kron
        self.ignore_subject = ignore_subject

    def vectorize(self, dependency_graph):
        nodes = dependency_graph.nodes

        if self.operator == 'head':
            assert len(nodes[0]['deps']) == 1

            head_address, = nodes[0]['deps']['ROOT']
            return self.node_to_vector(nodes[head_address])

        elif self.operator in ('add', 'mult'):
            if self.ignore_subject:
                raise ValueError(
                    'Can not ignore subject as {} is not grammar aware compositional method. '
                    'Do not use the --ignore_subject flag.'
                    ''.format(self.operator)
                )

            tokens = tuple((node['lemma'], node['tag']) for node in nodes.values() if node['address'])
            return getattr(self.space, self.operator)(*tokens)

        elif self.operator in self.transitive_operators:
            assert graph_signature(dependency_graph) == transitive_sentence(self.tagset)

            object_ = self.node_to_vector(nodes[3])
            if self.ignore_subject:
                subject = sparse.csr_matrix(matlib.ones(object_.shape))
            else:
                subject = self.node_to_vector(nodes[1])

            length = self.space.matrix.shape[1]

            if self.operator in ('kron', 'relational'):

                if self.operator == 'kron':
                    verb = self.node_to_vector(nodes[2])

                    if self.high_dim_kron:
                        return subject.toarray().flatten(), verb.toarray().flatten(), object_.toarray().flatten()

                    verb_matrix = sparse.kron(verb, verb, format='csr')
                else:
                    assert (length ** 2) == self.verb_space.matrix.shape[1]

                    try:
                        verb_matrix = self.node_to_vector(nodes[2], space=self.verb_space)
                    except KeyError:
                        logger.exception('Could not retrieve verb matrix for %s.', nodes[2]['lemma'])
                        verb_matrix = sparse.csr_matrix((1, length ** 2), dtype=float)

                subject_object = sparse.kron(subject, object_, format='csr')

                return verb_matrix.multiply(subject_object)

            else:
                assert (length ** 2) == self.verb_space.matrix.shape[1]

                try:
                    verb_matrix = self.node_to_vector(nodes[2], space=self.verb_space)
                except KeyError:
                    logger.exception('Could not retrieve verb matrix for %s.', nodes[2]['lemma'])
                    verb_matrix = sparse.csr_matrix((length, length), dtype=float)
                else:
                    verb_matrix = sparse.csr_matrix(
                        verb_matrix
                        .todense()
                        .reshape((length, length))
                    )

                def copy_object():
                    return subject.T.multiply(verb_matrix.dot(object_.T)).T

                def copy_subject():
                    return subject.dot(verb_matrix).multiply(object_)

                return {
                    'copy-object': copy_object,
                    'copy-subject': copy_subject,
                    'frobenious-add': lambda: copy_object() + copy_subject(),
                    'frobenious-mult': lambda: copy_object().multiply(copy_subject()),
                    'frobenious-outer': lambda: sparse.kron(copy_object(), copy_subject()),
                }[self.operator]()

        else:
            raise ValueError('Operator {} is not supported'.format(self.operator))

    def info(self):
        return '({s.BOLD}{operator}{s.RESET})'.format(
            s=colored.style,
            operator=self.operator,
        )

    def node_to_vector(self, node, space=None):
        if space is None:
            space = self.space
        try:
            return space[node['lemma'], node['tag']]
        except KeyError:
            if self.sloppy:
                logger.warning('Could not retrieve: %s %s', node['lemma'], node['tag'])
                return sparse.csr_matrix((1, space.matrix.shape[1]), dtype=float)
            else:
                raise


@command()
def similarity(
        pool,
        dataset,
        no_p11n,
        composition_operator,
        space,
        verb_space,
        output=('o', 'sentence_similarity.h5', 'Result output file.'),
        key=('', 'dataset', 'The key of the result in the output file.'),
        sloppy=('', False, 'Allow missing words.'),
        ignore_subject=('', False, 'Do not take subject into account, use the vector of ones instead.'),
):

    if dataset.vectorizer == 'lexical':
        assert composition_operator == 'head'

    # Kron is too slow for vector spaces with more than 3K dimensions, however,
    # it's possible to calculation innter product based similarity without doing
    # Kron products.
    high_dim_kron = (space.matrix.shape[1] > 3000) and composition_operator == 'kron'

    vectorizer = CompositionalVectorizer(
        space,
        composition_operator,
        tagset=dataset.tagset,
        verb_space=verb_space,
        sloppy=sloppy,
        high_dim_kron=high_dim_kron,
        ignore_subject=ignore_subject,
    )


    experiment = SimilarityExperiment(show_progress_bar=not no_p11n, pool=pool)

    experiment.evaluate(
        dataset=dataset,
        vectorizer=vectorizer,
        high_dim_kron=high_dim_kron,
    ).to_hdf(output, key=key)


@command()
def entailment_direction(
    dataset,
    space,
    pool,
    no_p11n,
    argument_counts=('', 'ANDailment_argument_counts.h5', ''),
    output=('o', 'sentence_similarity.h5', 'Result output file.'),
    key=('', 'dataset', 'The key of the result in the output file.'),
):
    argument_counts = pd.read_hdf(argument_counts, key='arguments')
    argument_counts.reset_index('argument', inplace=True)
    argument_counts.sort_index(inplace=True)

    EntailmentDirectionExperiment(show_progress_bar=not no_p11n, pool=pool).evaluate(
        dataset=dataset,
        space=space,
        argument_counts=argument_counts,
    ).to_hdf(output, key=key)


@command()
def entailment_direction_verb_object_vectors(
    dataset,
    argument_counts=('', 'ANDailment_argument_counts.h5', ''),
    argument_filter=('', '', ''),
):
    argument_counts = pd.read_hdf(argument_counts, key='arguments')

    if argument_filter:
        argument_counts = argument_counts.loc[(slice(None), slice(None), argument_filter), slice(None)]

    arguments = argument_counts.groupby(level=['relation', 'argument']).sum()['count']

    arguments = arguments[arguments.index.get_level_values('argument') != '']

    p_argument = arguments.groupby(level='argument').sum() / arguments.sum()
    assert p_argument.index.is_unique

    p_argument_given_verb = (
        arguments
        .groupby(level=('relation'))
        .apply(lambda df: df.reset_index('relation', drop=True) / df.sum())
    )

    def kl(verb):
        verb_arguments = p_argument_given_verb.loc[verb]
        return entropy(verb_arguments, p_argument.loc[verb_arguments.index])

    def n(verb):
        return arguments.loc[verb].sum()

    df = dataset.read_file()
    df = df.loc[~df['is_bidirectional'] & ~df['is_temporal']]

    df['kl_lhs'] = df['rule_lhs'].apply(kl)
    df['kl_rhs'] = df['rule_rhs'].apply(kl)

    df['n_lhs']= df['rule_lhs'].apply(n)
    df['n_rhs']= df['rule_rhs'].apply(n)

    for method, column_prefix in ('Frequency', 'n'), ('KL', 'kl'):

        correct = (df['{}_lhs'.format(column_prefix)] > df['{}_rhs'.format(column_prefix)]) == df['entails']
        accuracy = correct.sum() / len(correct)

        print('{}: accuracy {:.3f}, support {}, {argument_filter}.'.format(
                method,
                accuracy,
                len(correct),
                argument_filter=argument_filter,
            )
        )
