import pandas as pd

from colored import style
from scipy import stats
from sklearn.metrics import pairwise

from fowler.corpora.util import Worker


def cosine_similarity(s1, s2):
    return pairwise.cosine_similarity(s1, s2)[0][0]


def inner_product(s1, s2):
    return s1.multiply(s2).sum()


class SimilarityExperiment(Worker):
    def evaluate(self, dataset):

        result = pd.DataFrame.from_records(
            [
                (
                    cosine_similarity(s1, s2),
                    inner_product(s1, s2),
                )
                for s1, s2 in self.progressify(
                    dataset.pairs(),
                    description='Similarity',
                    max=len(dataset.dataset),
                )
            ],
            columns=('cos', 'inner_product'),
        )

        dataset.dataset['cos'] = result['cos']
        dataset.dataset['inner_product'] = result['inner_product']

        comparison = dataset.dataset[[dataset.human_judgement_column, 'cos']]
        comparison = comparison[comparison[dataset.human_judgement_column].notnull()]

        rho, p = stats.spearmanr(comparison)
        print(
            'Spearman correlation{info}, cosine): '
            '{style.BOLD}rho={rho:.3f}{style.RESET}, p={p:.5f}, support={support}'
            .format(
                rho=rho,
                p=p,
                style=style,
                info=dataset.info(),
                support=len(comparison),
            )
        )

        return dataset
