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


class SimilarityExperiment(Worker):

    def evaluate(self, dataset, vectorizer):
        pairs = list(dataset.dependency_graphs_pairs())

        # TODO: Refactor to mimic scikit-learn pipeline.
        sent_vectors = (
            (
                g1,
                vectorizer.vectorize(g1).toarray().flatten(),
                g2,
                vectorizer.vectorize(g2).toarray().flatten(),
                score,
            ) + tuple(extra)
            for g1, g2, score, *extra in pairs
        )

        result = pd.DataFrame.from_records(
            [
                (
                    tree(s1),
                    tree(s2),

                    1 / (1 + distance.euclidean(s1_vect, s2_vect)),
                    1 - distance.cosine(s1_vect, s2_vect),
                    1 - distance.correlation(s1_vect, s2_vect),
                    s1_vect.dot(s2_vect.T),

                    score,
                ) + tuple(extra)
                for s1, s1_vect, s2, s2_vect, score, *extra in self.progressify(
                    sent_vectors,
                    description='Similarity',
                    max=len(pairs)
                )
            ],
            columns=(
                'unit1',
                'unit2',

                'eucliedean',
                'cos',
                'correlation',
                'inner_product',

                'score',
            ) + getattr(dataset, 'extra_fields', tuple()),
        )

        if not result.notnull().all().all():
            logger.warning('Null values in similarity scores.')

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
