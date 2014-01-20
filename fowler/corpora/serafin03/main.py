"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import sys
import logging
from collections import deque
from itertools import chain

import pandas as pd
import numpy as np

from scipy.sparse import vstack, hstack, csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels

from nltk.tokenize import word_tokenize

from fowler.switchboard.swda import CorpusReader
from fowler.switchboard.util import get_conversation_ids

from fowler.corpora import util, models
from fowler.corpora.dispatcher import Dispatcher


CHUNK_SIZE = 10000


logger = logging.getLogger(__name__)


def utterance_tokens(utterance, ngram_len=1):
    ngram = deque(['<BEGIN>'], ngram_len)

    words = chain(word_tokenize(utterance.text), ['<END>'])

    for w in words:
        ngram.append(w)

        yield '_'.join(ngram)


def middleware_hook(kwargs, f_args):

    swda = kwargs['swda'] = CorpusReader(kwargs['swda'])
    if 'swda' not in f_args:
        del kwargs['swda']

    logger.info('Reading the utterances.')

    utterances = list(swda.iter_utterances(display_progress=False))

    train_split = get_conversation_ids(kwargs.pop('train_split'))
    test_split = get_conversation_ids(kwargs.pop('test_split'))

    train_utterances = [u for u in utterances if u.conversation_no in train_split]
    limit = kwargs.pop('limit')
    if limit:
            train_utterances = [u for u in utterances if u.conversation_no in train_split][:limit]
    kwargs['train_utterances'] = train_utterances
    test_utterances = kwargs['test_utterances'] = [u for u in utterances if u.conversation_no in test_split]

    assert train_utterances and test_utterances

    if 'space' in f_args:
        kwargs['space'] = models.read_space_from_file(kwargs['space'])
    else:
        del kwargs['space']

    kwargs['y_train'] = [u.damsl_act_tag() for u in train_utterances]
    kwargs['y_test'] = [u.damsl_act_tag() for u in test_utterances]


dispatcher = Dispatcher(
    middleware_hook=middleware_hook,
    globaloptions=(
        ('j', 'n_jobs', -1, 'The number of CPUs to use to do computations. -1 means all CPUs.'),
        ('f', 'n_folds', 3, 'The number of folds used for cross validation.'),
        ('', 'swda', './swda', 'The path to the Switchboard corpus.'),
        ('', 'train_split', 'downloads/switchboard/ws97-train-convs.list.txt', 'The training splits'),
        ('', 'test_split', 'downloads/switchboard/ws97-test-convs.list.txt', 'The testing splits'),
        ('', 'space', 'space.h5', 'The space file.'),
        ('', 'limit', 0, 'Number of train utterances.')
    ),
)
command = dispatcher.command


def document_word(args):
    i, u, dictionary, ngram_len = args
    js = dictionary.loc[utterance_tokens(u, ngram_len=ngram_len)].id

    result = []
    for j in js:
        if np.isfinite(j):
            result.append([i, j, 1])

    if not result:
        result = [[i, 0, 0]]

    return result


@command()
def plain_lsa(
    train_utterances,
    test_utterances,
    y_train,
    y_test,
    n_jobs,
    n_folds,
    templates_env,
    pool,
    ngram_len=('', 1, 'Length of the tokens (bigrams, ngrams).'),

):
    """Perform the Plain LSA method."""

    words = list(set(chain.from_iterable(utterance_tokens(u, ngram_len=ngram_len) for u in train_utterances)))
    dictionary = pd.DataFrame({'id': np.arange(len(words))}, index=words)

    def extract_features(utterances, shape):
        data = list(
            chain.from_iterable(
                pool.imap(
                    document_word,
                    ((i, u, dictionary, ngram_len) for i, u in enumerate(utterances)),
                    chunksize=CHUNK_SIZE,
                )
            )
        )

        features = pd.DataFrame(data, columns=('row', 'col', 'data'))

        data = features['data'].values
        row = features['row'].values
        col = features['col'].values

        return csr_matrix((data, (row, col)), shape=shape)

    logger.info('Extracting %d features from the training set', len(train_utterances))
    X_train = extract_features(train_utterances, shape=(len(train_utterances), len(words)))

    logger.info('Extracting %d features from the testing set', len(test_utterances))
    X_test = extract_features(test_utterances, shape=(len(test_utterances), len(words)))

    evaluate(X_train, X_test, y_train, y_test, templates_env, {}, n_folds, n_jobs, 'Serafin et al. 2003', pool)


def space_compose(args):
    u, composer = args
    return composer(*utterance_tokens(u))


@command()
def composition(
    train_utterances,
    test_utterances,
    y_train,
    y_test,
    pool,
    space,
    n_jobs,
    n_folds,
    templates_env,
    word_composition_operator=('', 'add', 'What operator use for compositon. [add|mult]'),
    concatinate_prev_utterace=('', False, 'Concatinate the vector of a current utterance wiht the vector of the precious utterance.'),
):

    if word_composition_operator == 'mult':
        composer = space.multiply
    else:
        composer = space.add

    def extract_features(utterances):

        logger.info('Extracting features.')
        X = pool.map(space_compose, ((u, composer) for u in utterances), chunksize=CHUNK_SIZE)
        logger.debug('Stacking %d rows.', len(X))
        X = vstack(X, format='csr')

        if concatinate_prev_utterace:
            logger.debug('Getting previous utterances.')
            # It is basically the same X, just shifted one row up.
            prev_X = vstack([csr_matrix((1, X.shape[1])), csr_matrix(X)[:-1]], format='csr')

            # Reset prev. utterance vectors to 0 for the first utterance in a conversation.
            prev_conversation_no = None
            for row, u in enumerate(utterances):
                conversation_no = u.conversation_no
                if conversation_no != prev_conversation_no:
                    prev_X.data[prev_X.indptr[row]:prev_X.indptr[row + 1]] = 0
                prev_conversation_no = conversation_no

            prev_X.eliminate_zeros()

            assert (X[0] == prev_X[1]).todense().all()

            logger.debug('Hstacking utterances with thier previous utterances.')
            X = hstack([X, prev_X], format='csr')

        return X

    X_train = extract_features(train_utterances)
    X_test = extract_features(test_utterances)

    evaluate(X_train, X_test, y_train, y_test, templates_env, {}, n_folds, n_jobs, 'Comp sem.', pool)


def evaluate(X_train, X_test, y_train, y_test, templates_env, store_metadata, n_folds, n_jobs, paper, pool):
    # tuned_parameters = {
    #     'svd__n_components': (50, ),
    #     'nn__n_neighbors': (1, 5),
    # }

    pipeline = Pipeline(
        [
            ('svd', TruncatedSVD(n_components=50)),
            ('nn', KNeighborsClassifier()),
        ]
    )

    # clf = GridSearchCV(
    #     pipeline,
    #     tuned_parameters,
    #     cv=n_folds,
    #     scoring='accuracy',
    #     n_jobs=n_jobs,
    # )
    clf = pipeline

    logger.info('Training.')
    clf.fit(X_train, y_train)

    logger.info('Predicting %d labels.', X_test.shape[0])
    y_predicted = clf.predict(X_test)

    prfs = precision_recall_fscore_support(y_test, y_predicted)

    util.display(
        templates_env.get_template('classification_report.rst').render(
            argv=' '.join(sys.argv) if not util.inside_ipython() else 'ipython',
            paper=paper,
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
