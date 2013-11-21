"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import sys

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

from fowler.corpora.main.options import Dispatcher

from .classifier import PlainLSA

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def plain_lsa(
    cooccurrence_matrix,
    labels,
    templates_env,
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

    prfs = precision_recall_fscore_support(y_test, clf.predict(X_test))
    print(
        templates_env.get_template('classification_report.rst').render(
            argv=' '.join(sys.argv),
            paper='Serafin et al. 2003',
            clf=clf,
            tprfs=zip(y_test, *prfs),
            p_avg=np.average(prfs[0], weights=prfs[3]),
            r_avg=np.average(prfs[1], weights=prfs[3]),
            f_avg=np.average(prfs[2], weights=prfs[3]),
            s_sum=np.sum(prfs[3]),
        )
    )
