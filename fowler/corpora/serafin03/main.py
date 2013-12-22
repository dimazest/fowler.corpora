"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import sys

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.multiclass import unique_labels

from jinja2 import Environment, PackageLoader

import fowler.corpora
from fowler.corpora import io
from fowler.corpora.dispatcher import Dispatcher

from .classifier import PlainLSA


def middleware_hook(kwargs, f_args):
    if kwargs['path'].endswith('.h5'):

        with pd.get_store(kwargs['path'], mode='r') as store:
            if 'cooccurrence_matrix' in f_args:
                kwargs['cooccurrence_matrix'] = io.load_cooccurrence_matrix(store)

            if 'labels' in f_args:
                kwargs['labels'] = io.load_labels(store)

            if 'templates_env' in f_args:
                kwargs['templates_env'] = Environment(
                    loader=PackageLoader(fowler.corpora.__name__, 'templates')
                )

            if 'store_metadata' in f_args:
                kwargs['store_metadata'] = store.get_storer('data').attrs.metadata

    if 'path' not in f_args:
        del kwargs['path']


dispatcher = Dispatcher(
    middleware_hook=middleware_hook,
    globaloptions=(
        ('p', 'path', 'out.h5', 'The path to the store hd5 file.'),
    ),
)
command = dispatcher.command


@command()
def plain_lsa(
    cooccurrence_matrix,
    labels,
    templates_env,
    store_metadata,
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
    y_predicted = clf.predict(X_test)
    prfs = precision_recall_fscore_support(y_test, y_predicted)

    display(
        templates_env.get_template('classification_report.rst').render(
            argv=' '.join(sys.argv) if not inside_ipython() else 'ipython',
            paper='Serafin et al. 2003',
            clf=clf,
            tprfs=zip(unique_labels(y_test, y_predicted), *prfs),
            p_avg=np.average(prfs[0], weights=prfs[3]),
            r_avg=np.average(prfs[1], weights=prfs[3]),
            f_avg=np.average(prfs[2], weights=prfs[3]),
            s_sum=np.sum(prfs[3]),
            store_metadata=store_metadata,
            accuracy=accuracy_score(y_test, y_predicted),
        )
    )


def inside_ipython():
    try:
        return __IPYTHON__
    except NameError:
        pass


def display(value):
    if inside_ipython():
        from IPython.display import display as ipython_display, HTML
        ipython_display(HTML(rst_to_html(value)))
    else:
        print(value)


def rst_to_html(value):
    from docutils.examples import html_body
    return html_body(value)

