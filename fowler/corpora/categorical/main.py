"""Categorical vector space creation."""
import logging

import pandas as pd

from progress.bar import Bar
from scipy import sparse

from fowler.corpora.dispatcher import Dispatcher, Resource, NewSpaceCreationMixin, SpaceMixin, DictionaryMixin
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
        return (
            pd.read_hdf(
                self.kwargs['transitive_verb_arguments'],
                key=self.dictionary_key,
            )
            [
                ['verb_stem', 'subj_stem', 'subj_tag', 'obj_stem', 'obj_tag', 'count']
            ]
            # Because we get rid of the verb, there might be multiple entries for a verb_stem!
            .groupby(('verb_stem', 'subj_stem', 'subj_tag', 'obj_stem', 'obj_tag'), as_index=False).sum()
            .set_index('verb_stem')
            .loc[self.verbs]
            .reset_index()
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
    space,
    transitive_verb_arguments,
    dictionary_key,
    pool,
    output=('o', 'space.h5', 'Output verb vector space.'),
    chunk_size=('', 10000, 'The length of the chunk.'),
):
    groups = transitive_verb_arguments.groupby(
        ['subj_stem', 'subj_tag', 'obj_stem', 'obj_tag'],
    )

    # groups = Bar(
    #     'Subject object Kronecker products',
    #     max=1,
    #     # max=len(transitive_verb_arguments[['subj_stem', 'subj_tag', 'obj_stem', 'obj_tag']]),
    #     suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
    # ).iter(groups)

    verb_tensors = {}

    for (subj_stem, subj_tag, obj_stem, obj_tag), group in groups:
        # There have to be at most one identical subject, object tuple per verb.
        assert len(group['verb_stem'].unique()) == len(group)

        subject_vector = space[subj_stem, subj_tag]
        object_vector = space[obj_stem, obj_tag]

        if not subject_vector.size:
            logger.warning('Subject %s %s is empty!', subj_stem, subj_tag)
            continue

        if not object_vector.size:
            logger.warning('Object %s %s is empty!', obj_stem, obj_tag)
            continue

        subject_object_tensor = sparse.kron(subject_vector, object_vector)

        for verb_stem, count in group[['verb_stem', 'count']].values:

            t = subject_object_tensor * count

            if verb_stem not in verb_tensors:
                verb_tensors[verb_stem] = t
            else:
                verb_tensors[verb_stem] = verb_tensors[verb_stem] + t

    import ipdb; ipdb.set_trace()
