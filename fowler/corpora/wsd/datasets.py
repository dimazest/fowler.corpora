import pandas as pd
from scipy.sparse import kron

from fowler.corpora.util import Worker


class Dataset(Worker):

    def to_hdf(self, file_path, key='dataset'):
        self.dataset.to_hdf(file_path, key=key)


class SimilarityDataset(Dataset):

    def __init__(
        self,
        dataset_filename,
        space,
        tagset,
        *args,
        human_judgement_column=None,
        group=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if human_judgement_column is not None:
            self.human_judgement_column = human_judgement_column

        self.dataset = self.read(
            dataset_filename,
            human_judgement_column,
            group,
            )
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

    def __init__(self, composition_operator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.composition_operator = composition_operator

    @classmethod
    def read(cls, dataset_filename, human_judgement_column=None, group=True):
        if human_judgement_column is None:
            human_judgement_column = cls.human_judgement_column

        df = pd.read_csv(
            dataset_filename,
            sep=cls.dataset_sep,
            usecols=cls.group_columns + (human_judgement_column, ),
        )

        if group:
            df = df.groupby(cls.group_columns, as_index=False).mean()

        # if self.google_vectors:
        #     df = df[grouped['verb2'] != 'emphasise']
        #     df = df[grouped['subject1'] != 'programme']
        #     df = df[grouped['subject2'] != 'programme']
        #
        # if self.context.limit:
        #     df = df.head(self.limit)

        return df

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
        else:
            verb_vectors = {v: V(v) for v in verbs}

        return verb_vectors

    def compose(self, subject, verb, object_):
        Sentence = {
            'verb': lambda subject, verb, object_: verb,
            'add': lambda subject, verb, object_: verb + subject + object_,
            'mult': lambda subject, verb, object_: verb.multiply(subject).multiply(object_),
            'kron': lambda subject, verb, object_: verb.multiply(kron(subject, object_, format='bsr')),
        }[self.composition_operator]

        return Sentence(subject, verb, object_)


class Tagger:
    # TODO: it would be cool to move it somewhere

    tag_mappings = {
        'bnc': {'N': 'SUBST', 'V': 'VERB'},
        'bnc+ccg': {'N': 'N', 'V': 'V'},
        'ukwac': {'N': 'N', 'V': 'V'},
    }

    def __init__(self, space, tagset):
        self.space = space
        self.with_tags = space.row_labels.index.nlevels == 2
        self.tagset = tagset

    def tag(self, w, tag):

        if (w, tag) == ('parish', 'J'):
            tag = 'N'
        if (w, tag) == ('raffle', 'J'):
            tag = 'N'

        if self.with_tags:
            tag = self.tag_mappings[self.tagset][tag]
            return self.space[w, tag]

        return self.space[w]

    def V(self, v):
        """Verb."""
        return self.tag(v, 'V')

    def N(self, v):
        """Noun (substantive)."""
        return self.tag(v, 'N')

    def A(self, v):
        """Adjective."""
        return self.tag(v, 'J')
