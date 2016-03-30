import logging
from itertools import chain

import pandas as pd

from colored import style
from scipy import stats
from scipy.spatial import distance

from fowler.corpora.util import Worker


logger = logging.getLogger(__name__)


def _tree(dg, i):
    node = dg.get_by_address(i)

    deps = sorted(chain.from_iterable(node['deps'].values()))

    return node['lemma'], node['tag'], tuple(_tree(dg, dep) for dep in deps)


def tree(dg):
        node = dg.root

        deps = sorted(chain.from_iterable(node['deps'].values()))
        return node['lemma'], node['tag'], tuple(_tree(dg, dep) for dep in deps)


def kl(p, q, alpha=0.99):
    return stats.entropy(p, alpha * q + (1 - alpha) * p)


class SimilarityExperiment(Worker):

    def evaluate(self, dataset, vectorizer, high_dim_kron=False):
        pairs = list(dataset.dependency_graphs_pairs())

        # TODO: Refactor to mimic scikit-learn pipeline.
        sent_vectors = (
            (
                g1,
                vectorizer.vectorize(g1),
                g2,
                vectorizer.vectorize(g2),
                score,
            ) + tuple(extra)
            for g1, g2, score, *extra in pairs
        )

        if not high_dim_kron:
            sent_vectors = (
                (g1, v1.toarray().flatten(), g2, v2.toarray().flatten(), score) + tuple(extra)
                for g1, v1, g2, v2, score, *extra in sent_vectors
            )
            result_values = (
                (
                    g1,
                    g2,

                    1 / (1 + distance.euclidean(v1, v2)),
                    1 - distance.cosine(v1, v2),
                    1 - distance.correlation(v1, v2),
                    v1.dot(v2.T),
                    kl(v1, v2),
                    kl(v2, v1),

                    score,
                ) + tuple(extra)
                for g1, v1, g2, v2, score, *extra in sent_vectors
            )
            result_columns = (
                    'euclidean',
                    'cos',
                    'correlation',
                    'inner_product',
                    'entropy_s1_s2',
                    'entropy_s2_s1',
                )
        else:
            result_values = (
                (
                    g1,
                    g2,

                    (v1 * s1).dot(v2 * s2).T * (v1 * o1).dot((v2 * o2).T),

                    score,
                ) + tuple(extra)
                for g1, (s1, v1, o1), g2, (s2, v2, o2), score, *extra in sent_vectors
            )
            result_columns = (
                    'inner_product',
                )

        result = pd.DataFrame.from_records(
            [
                (tree(g1), tree(g2)) + tuple(rest)
                for g1, g2, *rest in self.progressify(
                    result_values,
                    description='Similarity',
                    max=len(pairs),
                )
            ],
            columns=(
                ('unit1', 'unit2', ) + result_columns + ('score', ) + getattr(dataset, 'extra_fields', tuple())
            )
        )

        if not result.notnull().all().all():
            logger.warning('Null values in similarity scores.')

        for column in result_columns:

            rho, p = stats.spearmanr(result[[column, 'score']])
            print(
                'Spearman correlation {info}, {column}: '
                '{style.BOLD}rho={rho:.3f}{style.RESET}, p={p:.5f}, support={support}'
                .format(
                    rho=rho,
                    p=p,
                    style=style,
                    info=vectorizer.info(),
                    support=len(result),
                    column=column,
                )
            )

        return result
