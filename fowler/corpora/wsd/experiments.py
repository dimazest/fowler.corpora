import logging
from itertools import chain

import pandas as pd

from colored import style
from scipy import stats
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

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
                    stats.entropy(v1),
                    stats.entropy(v2),

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
                    'entropy_s1',
                    'entropy_s2',
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


class EntailmentDirectionExperiment(Worker):

    def verb_objects_avg_distance(self, verb, object_, argument_counts, space):
        verb_objects = argument_counts.loc[verb, 'objects']
        objects = set(space.row_labels.index.levels[0].intersection(verb_objects['argument']).drop_duplicates())
        verb_objects = verb_objects[verb_objects['argument'].isin(objects)]

        verb_objects = pd.DataFrame(
            [(r['argument'], r['count']) for _, r in verb_objects.iterrows() if (r['argument'], 'N') in space],
            columns=('argument', 'count'),
        )

        try:
            object_vector = space[object_, 'N']
        except KeyError:
            print('Could not retrieve the vector for {}.'.format(object_))
        else:

            for metric in 'correlation', 'cosine':
                distances = pairwise_distances(
                    space.get_target_rows(
                        *((o, 'N') for o in verb_objects['argument'])
                    ).todense(),
                    object_vector.todense(),
                    metric=metric,
                )

                distances = (distances.flatten() * verb_objects['count'] / verb_objects['count'].sum())

                yield metric, 'sum', distances.sum()
                yield metric, 'mean', distances.mean()


    def evaluate(self, dataset, space, argument_counts):
        pairs = list(dataset.dependency_graphs_pairs())

        verb_specificity = (
            (
                g1,
                self.verb_objects_avg_distance(
                    verb=g1.nodes[2]['lemma'],
                    object_=g1.nodes[3]['lemma'],
                    argument_counts=argument_counts,
                    space=space,
                ),
                g2,
                self.verb_objects_avg_distance(
                    verb=g2.nodes[2]['lemma'],
                    object_=g2.nodes[3]['lemma'],
                    argument_counts=argument_counts,
                    space=space,
                ),

                score,
            ) + tuple(extra)
            for g1, g2, score, *extra in pairs
        )


        results = (
            tuple((g1, g1_score, g2, g2_score, score) + tuple(extra) for g1_score, g2_score in zip(g1_scores, g2_scores))
            for g1, g1_scores, g2, g2_scores, score, *extra
            in self.progressify(
                verb_specificity,
                description='Entailment',
                max=len(pairs),
            )
        )

        results = [
            (tree(g1), tree(g2), similarity, aggregate_function, g1_score, g2_score, score) + tuple(extra)
            for g1, (similarity, aggregate_function, g1_score), g2, (_, _, g2_score), score, *extra
            in chain.from_iterable(results)
        ]

        results = pd.DataFrame.from_records(
            results,
            columns=(
                'unit1',
                'unit2',
                'distance',
                'aggregator',
                'v1_diversity',
                'v2_diversity',
                'score',
            ) + getattr(dataset, 'extra_fields', tuple())
        )

        return results