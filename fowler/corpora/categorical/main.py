"""Categorical vector space creation."""
import logging
import pickle

import numpy as np
import pandas as pd

from progress.bar import Bar
from scipy import sparse

from fowler.corpora.dispatcher import Dispatcher, Resource, NewSpaceCreationMixin, SpaceMixin, DictionaryMixin
from fowler.corpora.execnet import verb_space_builder
from fowler.corpora.models import Space
from fowler.corpora.space.util import read_tokens

logger = logging.getLogger(__name__)


class CategoricalDispatcher(Dispatcher, NewSpaceCreationMixin, SpaceMixin, DictionaryMixin):

    global__transitive_verb_arguments = '', 'transitive_verbs.h5', 'Counted transitive verbs '
    global__verbs = '', 'verbs.csv', 'List of verbs to build matrices for.'

    @Resource
    def verbs(self):
        return read_tokens(self.kwargs['verbs']).index

    @Resource
    def transitive_verb_arguments(self):
        """Counted transitive verbs together with their subjects and objects."""
        df = self.read_vso_file(
            self.kwargs['transitive_verb_arguments'],
            self.dictionary_key,
        )

        df.set_index('verb_stem', inplace=True)
        df = df.loc[self.verbs]
        df.index.names = 'verb_stem',  # index name should not be lost in the line above!

        df = df.reset_index()[
            ['verb_stem', 'verb_tag', 'subj_stem', 'subj_tag', 'obj_stem', 'obj_tag', 'count']
        ]

        df = df.groupby(
            ('verb_stem', 'verb_tag', 'subj_stem', 'subj_tag', 'obj_stem', 'obj_tag'),
            as_index=False,
        ).sum()

        return df

        #     [
        #         ['verb_stem', 'verb_tag', 'subj_stem', 'subj_tag', 'obj_stem', 'obj_tag', 'count']
        #     ]
        #     # Because we get rid of the verb, there might be multiple entries for a verb_stem!
        #     .groupby(('verb_stem', 'verb_tag', 'subj_stem', 'subj_tag', 'obj_stem', 'obj_tag'), as_index=False).sum()
        #     .set_index('verb_stem')
        #     .loc[self.verbs]
        #     .reset_index()
        # )

    @staticmethod
    def read_vso_file(path, key):
        return (
            pd.read_hdf(path, key=key)
        )

    @Resource
    def subjects(self):
        return (
            self
            .transitive_verb_arguments[['subj_stem', 'subj_tag']]
            .drop_duplicates()
            .rename(columns={'subj_stem': 'ngram', 'subj_tag': 'tag'})
        )

    @Resource
    def objects(self):
        return (
            self
            .transitive_verb_arguments[['obj_stem', 'obj_tag']]
            .drop_duplicates()
            .rename(columns={'obj_stem': 'ngram', 'obj_tag': 'tag'})
        )

    @Resource
    def subjects_objects(self):
        return pd.concat((self.subjects, self.objects)).drop_duplicates()


dispatcher = CategoricalDispatcher()
command = dispatcher.command


@command()
def extract_verb_arguments(
    subjects_objects,
    output=('o', 'targets.csv', 'File where targets should be written.'),
    lowercase=('', False, 'Lowercase tokens.'),
):
    if lowercase:
        subjects_objects['ngram'] = subjects_objects['ngram'].str.lower()
        subjects_objects.drop_duplicates(inplace=True)

    subjects_objects.to_csv(output, index=False)


@command()
def transitive_verb_space(
    space_file,
    transitive_verb_arguments,
    execnet_hub,
    output=('o', 'space.h5', 'Output verb vector space.'),
    chunk_size=('', 100, 'The length of a chunk.'),
):

    data_to_send = (
        'data',
        pickle.dumps(
            {
                'space_file': space_file,
            },
        )
    )

    def init(channel):
        channel.send(data_to_send)

    groups = transitive_verb_arguments.groupby(
        ['verb_stem', 'verb_tag']
    )

    groups = Bar(
        'Subject object Kronecker products',
        max=len(groups),
        suffix='%(index)d/%(max)d',
    ).iter(
        pickle.dumps(g) for g in groups
    )

    results = execnet_hub.run(
        remote_func=verb_space_builder,
        iterable=groups,
        init_func=init,
        verbose=False,
    )

    result = next(results)

    for r in results:
        for k, v in r.items():
            if k in result:
                result[k] += v
            else:
                result[k] = v

    result = list(result.items())

    verb_labels = [l for l, _ in result]
    verb_vectors = [v for _, v in result]

    del result

    matrix = sparse.vstack(verb_vectors)
    del verb_vectors

    row_labels = pd.DataFrame(
        {
            'ngram': [l[0] for l in verb_labels],
            'tag': [l[1] for l in verb_labels],
            'id': [i for i, _ in enumerate(verb_labels)],
        }
    ).set_index(['ngram', 'tag'])

    column_labels = pd.DataFrame(
        {
            'ngram': list(range(matrix.shape[1])),
            'tag': list(range(matrix.shape[1])),
            'id': list(range(matrix.shape[1])),
        }
    ).set_index(['ngram', 'tag'])

    space = Space(
        matrix,
        row_labels=row_labels,
        column_labels=column_labels,
    )

    space.write(output)


@command()
def relgrams_verb_space(
    space,
    relgrams_relation_arguments=('', 'ANDailment_argument_counts.csv', ''),
    output=('o', 'space.h5', 'Output verb vector space.'),
):
    relgrams_relation_arguments = pd.read_csv(relgrams_relation_arguments, index_col='relation')

    groups = relgrams_relation_arguments.groupby(level='relation')

    def iter_arguments(df, kind):
        arguments = df[df['argument_type'] == kind]

        for _, (argument, count) in arguments[['argument', 'count']].iterrows():
            try:
                yield space[argument, 'N'] * count
            except (KeyError, TypeError):
                continue

    result = {}
    for relation, arguments in groups:

        print(relation)

        subject = np.sum(iter_arguments(arguments, 'subjects'))
        object_ = np.sum(iter_arguments(arguments, 'objects'))

        assert not isinstance(subject, float)

        result[relation] = sparse.kron(subject, object_)

    result = list(result.items())

    # A hack to make sure that all the vectors have the same shape.
    verb_shape = result[0][1].shape

    verb_labels = [l for l, v in result if v.shape == verb_shape]
    verb_vectors = [v for _, v in result if v.shape == verb_shape]

    matrix = sparse.vstack(verb_vectors)

    row_labels = pd.DataFrame(
        {
            'ngram': verb_labels,
            'tag': 'V',  # TODO: figure out real tag.
            'id': [i for i, _ in enumerate(verb_labels)],
        }
    ).set_index(['ngram', 'tag'])

    column_labels = pd.DataFrame(
        {
            'ngram': list(range(matrix.shape[1])),
            'tag': list(range(matrix.shape[1])),
            'id': list(range(matrix.shape[1])),
        }
    ).set_index(['ngram', 'tag'])

    space = Space(
        matrix,
        row_labels=row_labels,
        column_labels=column_labels,
    )

    space.write(output)
