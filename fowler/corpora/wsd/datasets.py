import logging

import pandas as pd

from scipy.sparse import kron, csr_matrix

from fowler.corpora.util import Worker


logger = logging.getLogger(__name__)


tag_mappings = {
    'bnc': {'N': 'SUBST', 'V': 'VERB', 'J': 'ADJ'},
    'bnc-ccg': {'N': 'N', 'V': 'V', 'J': 'J'},
    'ukwac': {'N': 'N', 'V': 'V', 'J': 'J'},
    # TODO: Is Brown mappting correct?
    # TODO: What do we do with adverbs?
    'brown': {'N': 'N', 'V': 'V', 'J': 'R'},
}


class Dataset(Worker):

    @classmethod
    def read(cls, dataset_filename):
        df = pd.read_csv(
            dataset_filename,
            sep=cls.dataset_sep,
            usecols=cls.group_columns + (cls.human_judgement_column, ),
        )

        if getattr(cls, 'group', False):
            df = df.groupby(cls.group_columns, as_index=False).mean()

        # if self.google_vectors:
        #     df = df[grouped['verb2'] != 'emphasise']
        #     df = df[grouped['subject1'] != 'programme']
        #     df = df[grouped['subject2'] != 'programme']
        #
        # if self.context.limit:
        #     df = df.head(self.limit)

        return df

    def info(self):
        return ''


class SimilarityDataset(Dataset):

    def __init__(
        self,
        dataset_filename,
        space,
        tagset,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset = self.read(dataset_filename)
        self.space = space
        self.tagset = tagset


class KS13Dataset(SimilarityDataset):
    """Paraphrasing dataset provided by [1] and re-scored by [2].

    [1] Mitchell, J., & Lapata, M. (2010). Composition in distributional
        models of semantics. Cognitive science, 34(8), 1388-1429.

    [2] Kartsaklis, Dimitri, and Mehrnoosh Sadrzadeh. "Prior Disambiguation of
        Word Tensors for Constructing Sentence Vectors."
        In EMNLP, pp. 1590-1601. 2013.

    """

    dataset_sep = ' '
    verb_columns = 'verb1', 'verb2'
    group_columns = 'subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2'
    human_judgement_column = 'score'
    group = True

    def __init__(
        self,
        composition_operator,
        verb_space,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.composition_operator = composition_operator
        self.verb_space = verb_space

    def pairs(self):
        verb_vectors = self.verb_vectors()

        t = Tagger(self.space, self.tagset)
        V = lambda w: verb_vectors[w]
        N = t.N

        sentence_word_vectors = (
            ((N(s1), V(v1), N(o1)), (N(s2), V(v2), N(o2)))
            for s1, v1, o1, s2, v2, o2 in self.dataset[list(self.group_columns)].values
        )

        return (
            (self.compose(*sent1), self.compose(*sent2))
            for sent1, sent2 in sentence_word_vectors
        )

    def verb_vectors(self):
        verbs = pd.concat([self.dataset[vc] for vc in self.verb_columns]).unique()

        t = Tagger(self.space, self.tagset)
        V = t.V

        if self.composition_operator == 'kron':
            verb_vectors = dict(
                zip(
                    verbs,
                    self.progressify(
                        (kron(V(v), V(v)) for v in verbs),
                        description='Verb matrices',
                        max=len(verbs)
                    )
                )
            )
        elif self.composition_operator in (
            'relational',
            'copy-object',
            'copy-subject',
            'frobenious-add',
            'frobenious-mult',
            'frobenious-outer',
        ):
            # Verb matrix extractor as a vector
            verb_space_tagger = Tagger(self.verb_space, self.tagset)
            VV = verb_space_tagger.V

            if self.composition_operator == 'relational':
                verb_vectors = {v: VV(v) for v in verbs}
            else:
                length = self.space.matrix.shape[1]
                assert (length ** 2) == VV(verbs[0]).shape[1]

                def M(v):
                    return csr_matrix(
                        VV(v)
                        .todense()
                        .reshape((length, length))
                    )

                verb_vectors = {v: M(v) for v in verbs}
        else:
            verb_vectors = {v: V(v) for v in verbs}

        return verb_vectors


# TODO: remove.
class Tagger:

    def __init__(self, space, tagset):
        self.space = space
        self.with_tags = space.row_labels.index.nlevels == 2
        self.tagset = tagset

    def tag(self, w, tag):

        if (w, tag) == ('parish', 'J'):
            logger.warning('Changing <parish, J> to <parish, N>')
            tag = 'N'
        if (w, tag) == ('raffle', 'J'):
            logger.warning('Changing <raffle, J> to <parish, N>')
            tag = 'N'

        if self.with_tags:
            tag = tag_mappings[self.tagset][tag]
            return self.space[w, tag]

        return self.space[w]

    def V(self, v):
        """Verb."""
        return self.tag(v, 'V')

    def N(self, v):
        """Noun (substantive)."""
        return self.tag(v, 'N')

    def J(self, v):
        """Adjective."""
        return self.tag(v, 'J')
