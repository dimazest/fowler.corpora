import logging

import pandas as pd

from colored import style
from scipy import stats
from sklearn.metrics import pairwise

from fowler.corpora.util import Worker


logger = logging.getLogger(__name__)


def cosine_similarity(s1, s2):
    return pairwise.cosine_similarity(s1, s2)[0][0]


def inner_product(s1, s2):
    return s1.dot(s2.T)[0, 0]


class SimilarityExperiment(Worker):
    def evaluate(self, dataset, vectorizer):
        pairs = list(dataset.dependency_graphs_pairs())

        # TODO: Refactor to mimic scikit-learn pipeline.
        sent_vectors = (
            (g1, vectorizer.vectorize(g1), g2, vectorizer.vectorize(g2), score) + tuple(extra)
            for g1, g2, score, *extra in pairs
        )

        result = pd.DataFrame.from_records(
            [
                (
                    str(s1.tree()),
                    str(s2.tree()),
                    cosine_similarity(s1_vect, s2_vect),
                    inner_product(s1_vect, s2_vect),
                    score,
                ) + tuple(extra)
                for s1, s1_vect, s2, s2_vect, score, *extra in self.progressify(
                    sent_vectors,
                    description='Similarity',
                    max=len(pairs)
                )
            ],
            columns=(
                'unit1', 'unit2',
                'cos', 'inner_product', 'score',
            ) + getattr(dataset, 'extra_fields', tuple()),
        )

        rho, p = stats.spearmanr(result[['cos', 'score']])
        print(
            'Spearman correlation {info}, cosine): '
            '{style.BOLD}rho={rho:.3f}{style.RESET}, p={p:.5f}, support={support}'
            .format(
                rho=rho,
                p=p,
                style=style,
                info=vectorizer.info(),
                support=len(result),
            )
        )

        return result
