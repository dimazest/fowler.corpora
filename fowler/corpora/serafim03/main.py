"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import sys

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from fowler.corpora.main.options import Dispatcher

from .classifier import PlainLSA

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def plain_lsa(
    cooccurrence_matrix,
    labels,
    k=('k', 50, 'The number of dimensions after SVD applicaion.'),
    n_jobs=('j', -1, 'The number of CPUs to use to do computations. -1 means all CPUs.'),
    n_folds=('f', 10, 'The number of folds used for cross validation.'),
):
    """Perform the Plain LSA method."""
    X_train, X_test, y_train, y_test = train_test_split(
        cooccurrence_matrix.T,
        labels,
        test_size=0.5,
        random_state=0,
    )

    tuned_parameters = [
        {'k': [3, 10, 40, 50, 60, 100]},
    ]

    clf = GridSearchCV(
        PlainLSA(),
        tuned_parameters,
        cv=n_folds,
        scoring='accuracy',
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)

    grid_scores = '\n'.join(
        '{s.mean_validation_score:0.3f} '
        '(+/-{pm:0.03f}) '
        'for {s.parameters}'
        ''.format(
            s=s,
            pm=s.cv_validation_scores.std() / 2.0,
        ) for s in clf.grid_scores_
    )

    print_results(
        paper='Serafin et al. 2003',
        clf=clf,
        grid_scores=grid_scores,
        classification_report=classification_report(y_test, clf.predict(X_test)),
    )


def print_results(**context):
    c = {'argv': ' '.join(sys.argv)}
    c.update(context)

    print(
        'Hyper parameter estimation\n'
        '--------------------------\n'
        '\n'
        ':paper: {paper}\n'
        ':command: {argv}\n'
        '\n'
        'Best parameters set found on development set: *{clf.best_estimator_}*\n'
        'Grid accuracy scores on development set\n'
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        '{grid_scores}\n'
        '\n'
        'Evaluation results\n'
        '------------------\n'
        '\n'
        '{classification_report}\n'
        'The model is trained on the full development set.\n'
        'The scores are computed on the full evaluation set.\n'
        '\n'
        ''.format(**c)
    )



