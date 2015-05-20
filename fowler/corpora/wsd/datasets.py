import logging

import pandas as pd

from colored import style
from scipy.sparse import kron, csr_matrix

from fowler.corpora.util import Worker


logger = logging.getLogger(__name__)


tag_mappings = {
    'bnc': {'N': 'SUBST', 'V': 'VERB'},
    'bnc+ccg': {'N': 'N', 'V': 'V', 'J': 'J'},
    'ukwac': {'N': 'N', 'V': 'V'},
}


class Dataset(Worker):

    def to_hdf(self, file_path, key='dataset'):
        self.dataset.to_hdf(file_path, key=key)


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

    def compose(self, subject, verb, object_):

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

        }[self.composition_operator]

        return Sentence()

    @classmethod
    def tokens(cls, df, tagset):
        def extract_tokens(series, tag=None):
            series = frame.unique()
            result = pd.DataFrame({'ngram': series})

            if tag is not None:
                result['tag'] = tag

            return result

        tm = tag_mappings[tagset]

        return (
           pd
           .concat(
               [
                   extract_tokens(df[c], t)
                   for c, t in (
                       ('verb1', tm['V']),
                       ('subject1', tm['N']),
                       ('object1', tm['N']),
                       ('verb2', tm['V']),
                       ('subject2', tm['N']),
                       ('object2', tm['N']),
                   )
               ]
            )
        )

    def info(self):
        return ' ({style.BOLD}{co}{style.RESET})'.format(
            style=style,
            co=self.composition_operator,
        )

class SimLex999Dataset(SimilarityDataset):
    """SimLex-999 is a gold standard resource for the evaluation of models that
       learn the meaning of words and concepts [1].

    [1] SimLex-999: Evaluating Semantic Models with (Genuine) Similarity
        Estimation. 2014. Felix Hill, Roi Reichart and Anna Korhonen. Preprint
        pubslished on arXiv. arXiv:1408.3456

        http://www.cl.cam.ac.uk/~fh295/simlex.html

    """

    dataset_sep = '\t'
    group_columns = 'word1', 'word2', 'POS'
    human_judgement_column = 'SimLex999'

    @classmethod
    def tokens(cls, df, tagset):
        result = pd.concat(
            [
                df[['word1', 'POS']].rename(columns={'word1': 'ngram', 'POS': 'tag'}),
                df[['word2', 'POS']].rename(columns={'word2': 'ngram', 'POS': 'tag'}),
            ]
        )

        result.loc[result['tag'] == 'N', 'tag'] = tag_mappings[tagset]['N']
        result.loc[result['tag'] == 'V', 'tag'] = tag_mappings[tagset]['V']
        result.loc[result['tag'] == 'A', 'tag'] = tag_mappings[tagset]['J']

        return result

    def pairs(self):
        t = Tagger(self.space, self.tagset)

        m = {
            'A': t.J,
            'V': t.V,
            'N': t.N
        }

        word_vector_pairs = (
            (m[tag](w1), m[tag](w2))
            for w1, w2, tag in self.dataset[list(self.group_columns)].values
        )

        return word_vector_pairs


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


dataset_types = {
    'ks13': KS13Dataset,
    'simlex999': SimLex999Dataset,
}
