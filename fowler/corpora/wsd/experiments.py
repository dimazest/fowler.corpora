from colored import style
from scipy import stats
from sklearn.metrics import pairwise

from fowler.corpora.util import Worker


def cosine_similarity(s1, s2):
    return pairwise.cosine_similarity(s1, s2)[0][0]


class SimilarityExperiment(Worker):
    def evaluate(self, dataset, composition_operator):
        dataset.dataset['cos'] = list(
            self.progressify(
                (cosine_similarity(s1, s2) for s1, s2 in dataset.pairs()),
                description='Cosine similarity',
                max=len(dataset.dataset),
            )
        )

        comparison = dataset.dataset[[dataset.human_judgement_column, 'cos']]
        comparison = comparison[comparison[dataset.human_judgement_column].notnull()]

        rho, p = stats.spearmanr(comparison)
        print(
            'Spearman correlation ({style.BOLD}{co}{style.RESET}, cosine): '
            '{style.BOLD}rho={rho:.3f}{style.RESET}, p={p:.5f}, support={support}'
            .format(
                rho=rho,
                p=p,
                style=style,
                co=composition_operator,
                support=len(comparison),
            )
        )

        return dataset
