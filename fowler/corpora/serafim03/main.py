"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import sys

from sklearn import cross_validation

from fowler.corpora.main.options import Dispatcher

from .classifier import PlainLSA

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def plain_lsa(
    cooccurrence_matrix,
    labels,
    k=('k', 50, 'The number of dimensions after SVD applicaion.'),
    n_jobs=('j', -1, 'The number of CPUs to use to do the computation. -1 means all CPUs.'),
    n_folds=('f', 10, 'The number of folds used for cross validation.'),
):
    """Perform the Plain LSA method."""
    classifier = PlainLSA(k)
    scores = cross_validation.cross_val_score(
        classifier,
        cooccurrence_matrix.T,
        labels,
        cv=n_folds,
        n_jobs=n_jobs,
    )

    print(
        "Paper: Serafin et al. '03.\n"
        'Command: {argv}\n'
        'Classifier: {classifier}\n'
        'Evaluation parameters: {items} items, {folds} folds\n'
        'Accuracy: average {:0.2%} (+/- {:0.2%}), max {:0.2%}'
        ''.format(
            scores.mean(),
            scores.std() * 2,
            scores.max(),
            classifier=classifier,
            items=len(labels),
            folds=n_folds,
            argv=' '.join(sys.argv),
            )
    )


