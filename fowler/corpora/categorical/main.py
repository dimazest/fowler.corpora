"""Categorical vector space creation."""

import numpy as np
import pandas as pd

from fowler.corpora.dispatcher import Dispatcher, Resource, NewSpaceCreationMixin, SpaceMixin, DictionaryMixin
from fowler.corpora.space.util import read_tokens


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
            [['verb_stem', 'subj_stem', 'obj_stem', 'count']]
            .set_index('verb_stem')
            .loc[self.verbs]
            .reset_index()
        )

    @Resource
    def subjects(self):
        return self.transitive_verb_arguments['subj_stem'].unique()

    @Resource
    def objects(self):
        return self.transitive_verb_arguments['obj_stem'].unique()

    @Resource
    def subjects_objects(self):
        return pd.DataFrame(
            {
                'ngram': np.unique(np.concatenate((self.subjects, self.objects))),
            }
        )

dispatcher = CategoricalDispatcher()
command = dispatcher.command


@command()
def extract_verb_arguments(
    subjects_objects,
    output=('o', 'targets.csv', 'File where targets should be written.'),
    lowercase=('', False, 'Lowercase tokens.'),
    pos_tag=('', 'SUBST', 'The part of speech tag.')
):
    if lowercase:
        subjects_objects['ngram'] = subjects_objects['ngram'].str.lower()
        subjects_objects.drop_duplicates(inplace=True)

    if pos_tag:
        subjects_objects['tag'] = pos_tag

    subjects_objects.to_csv(output, index=False)


@command()
def transitive_verbs_space(
    space,
    subjects,
    objects,
):

    import ipdb; ipdb.set_trace()
