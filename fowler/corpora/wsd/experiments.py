import pandas as pd

from colored import style
from scipy import stats
from sklearn.metrics import pairwise

from fowler.corpora.util import Worker


def cosine_similarity(s1, s2):
    return pairwise.cosine_similarity(s1, s2)[0][0]


def inner_product(s1, s2):
    # TODO: use numpy.
    return s1.multiply(s2).sum()


class SimilarityExperiment(Worker):
    def evaluate(self, dataset, composer):
        pairs = list(dataset.sentence_similarity_pairs())

        sent_vectors = (
            (s1, composer.compose(s1), s2, composer.compose(s2), score)
            for s1, s2, score in pairs
        )

        result = pd.DataFrame.from_records(
            [
                (
                    s1[0][0], s1[1][0], s1[2][0],
                    s2[0][0], s2[1][0], s2[2][0],
                    cosine_similarity(s1_vect, s2_vect),
                    inner_product(s1_vect, s2_vect),
                    score,
                )
                for s1, s1_vect, s2, s2_vect, score in self.progressify(
                    sent_vectors,
                    description='Similarity',
                    max=len(pairs)
                )
            ],
            columns=(
                'subject1', 'verb1', 'object1',
                'subject2', 'verb2', 'object2',
                'cos', 'inner_product', 'score',
            ),
        )

        rho, p = stats.spearmanr(result[['cos', 'score']])
        print(
            'Spearman correlation{info}, cosine): '
            '{style.BOLD}rho={rho:.3f}{style.RESET}, p={p:.5f}, support={support}'
            .format(
                rho=rho,
                p=p,
                style=style,
                info=composer.info(),
                support=len(result),
            )
        )

        return result
