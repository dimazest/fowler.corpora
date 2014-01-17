"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import sys
import logging
from itertools import islice

import pandas as pd
import numpy as np

from scipy.sparse import vstack

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels

from fowler.switchboard.swda import CorpusReader

from fowler.corpora import io, util, models
from fowler.corpora.dispatcher import Dispatcher


logger = logging.getLogger(__name__)


def middleware_hook(kwargs, f_args):

    # Cleanup: this is an old way to access space data.
    if 'cooccurrence_matrix' in f_args:

        with pd.get_store(kwargs['path'], mode='r') as store:
            if 'cooccurrence_matrix' in f_args:
                kwargs['cooccurrence_matrix'] = io.load_cooccurrence_matrix(store).T

            if 'labels' in f_args:
                kwargs['labels'] = io.load_labels(store)

            if 'store_metadata' in f_args:
                kwargs['store_metadata'] = store.get_storer('data').attrs.metadata
    # end cleanup.

    if 'swda' in f_args:
        kwargs['swda'] = CorpusReader(kwargs['swda'])

    if 'space' in f_args:
        kwargs['space'] = models.read_space_from_file(kwargs['path'])

    if 'path' not in f_args:
        del kwargs['path']


dispatcher = Dispatcher(
    middleware_hook=middleware_hook,
    globaloptions=(
        ('p', 'path', 'out.h5', 'The path to the store hd5 file.'),
        ('j', 'n_jobs', -1, 'The number of CPUs to use to do computations. -1 means all CPUs.'),
        ('f', 'n_folds', 10, 'The number of folds used for cross validation.'),

    ),
)
command = dispatcher.command


@command()
def plain_lsa(
    cooccurrence_matrix,
    labels,
    templates_env,
    store_metadata,
    n_jobs,
    n_folds,
):
    """Perform the Plain LSA method."""
    evaluate(cooccurrence_matrix, labels, templates_env, store_metadata, n_folds, n_jobs)


@command()
def composition(
    space,
    n_jobs,
    n_folds,
    templates_env,
    swda=('', './swda', 'The path to the Switchboard corpus.'),
    limit=('l', 0, 'Number of utterances.'),
    word_compsition_operator=('', 'add', 'The operator to be used to compose word vectors to obtain utterance vectors.'),
):
    utterances = swda.iter_utterances(display_progress=False)
    if limit:
        utterances = islice(utterances, limit)
    utterances = list(utterances)

    logging.info('Utterances are read.')

    labels = [u.damsl_act_tag() for u in utterances]

    if word_compsition_operator == 'add':
        compose = space.add
    else:
        compose = space.mult

    composed = [compose(*u.pos_words()) for u in utterances]
    composed = vstack(composed, format='csr')

    logger.info('The vectors for utterances are composed.')

    evaluate(composed, labels, templates_env, {}, n_folds, n_jobs)


def evaluate(cooccurrence_matrix, labels, templates_env, store_metadata, n_folds, n_jobs):
    X_train, X_test, y_train, y_test = train_test_split(
        cooccurrence_matrix,
        labels,
        test_size=0.5,
        random_state=0,
    )

    tuned_parameters = {
        'svd__n_components': (50, ),
        'nn__n_neighbors': (1, 5, 20, 40),
    }

    pipeline = Pipeline(
        [
            ('svd', TruncatedSVD()),
            ('nn', KNeighborsClassifier()),
        ]
    )

    clf = GridSearchCV(
        pipeline,
        tuned_parameters,
        cv=n_folds,
        scoring='accuracy',
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    prfs = precision_recall_fscore_support(y_test, y_predicted)

    util.display(
        templates_env.get_template('classification_report.rst').render(
            argv=' '.join(sys.argv) if not util.inside_ipython() else 'ipython',
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
