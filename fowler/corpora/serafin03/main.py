"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import logging
import re
import sys
from collections import Counter
from itertools import chain

import pandas as pd
import numpy as np

from scipy.sparse import vstack, hstack, csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels

from nltk import ngrams
from nltk.tokenize import word_tokenize

from fowler.switchboard.swda import CorpusReader
from fowler.switchboard.util import get_conversation_ids

from fowler.corpora import util, models, dispatcher


CHUNK_SIZE = 10000


logger = logging.getLogger(__name__)


class U:

    def __init__(self, utterance, *, lemmatize, pos):
        self._damsl_act_tag = utterance.damsl_act_tag()
        self.caller = utterance.caller
        self.conversation_no = utterance.conversation_no

        tags = {
            ',': 'PUN',  # ,
            '.': 'PUN',  # .
            ':': 'POS',  # --: -- {F oh, } I guess we usually enjoy ]  a good seafood restaurant.  /
            'a': 'ADJ',  # intriguing
            'bes': 'VERB',  # 's
            'cc': 'CONJ',  # and
            'dt': 'ART',  # a
            'hvs': 'VERB',  # 's: we find something we  like,  {F uh, } like  cashew shrimp or  something that's got a good, {F uh, } at a  particular restaurant  /
            'in': 'PREP',  # with
            'md': 'VERB',  # will: {C And } the other is with my whole  family whom we, {F uh, } go somewhere that the kids will enjoy <breathing>.  /
            'n': 'SUBST',  # food
            'pos': 'UNC',  # 's: {F Uh, } recently we have been hitting Pancho's up.  /
            'prp$': 'PRON',  # my: {C And } the other is with my whole  family whom we, {F uh, } go somewhere that the kids will enjoy <breathing>.  /
            'prp': 'PRON',  # you, it
            'r': 'ADV',  # very
            'rp': 'ADV',  # up: {F Uh, } recently we have been hitting Pancho's up.  /
            'to': 'PREP',  # to: [ we use to + ] go up there [ on, +
            'uh': '__uh__',  # So, um
            'v': 'VERB',  # be, talk
            'wdt': 'CONJ',  # that: we find something we  like,  {F uh, } like  cashew shrimp or  something that's got a good, {F uh, } at a  particular restaurant  /
            'wdt': 'CONJ',  # that: {C but } I'll just make one comment that, {F uh, } I just  retired from Penn State,  /
            'wp': 'CONJ',  # What: What about you? /
            'wp': 'PRON',  # what: {D Well, } that's what I worked on. /
            'wrb': 'CONJ',  # when: I used to go there when I was in college. /
            'xx': '__xx__',  # MUMBLEx: ((   ))  because they have


            '^jj': '__^jj__',  # mill: I guess one of the things we've, {F uh, }  started avoiding is the, {F uh, } run of the mill chop suey and things like that. /
            'cd': 'ADJ',  # One: One of [ [ th-, +  th-, ] +  this ]  book I have is called CHINESE COOKING MADE EASY.  /
            'gw': 'ART',  # the: I guess one of the things we've, {F uh, }  started avoiding is the, {F uh, } run of the mill chop suey and things like that. /


            'ex': 'PRON',  # there: <Breathing> {D Well, } there is two kinds.
            '^vbn': 'ADJ',  # crowded: it was over crowded


            '^nn': 'ADV',  # up: at first we were going to get a pick up truck,  with a camper on the back [ of it, +
            'pdt': 'ADJ',  # All: All these people were, - /

        }

        def T(t, w):
            try:
                return tags[t]
            except KeyError:
                print("            '{}': '',  # {}: {u.text}".format(t, w, u=utterance))

        self.words = tuple((w, T(t, w))for w, t in utterance.pos_lemmas(wn_lemmatize=lemmatize))
        if not pos:
            self.words = tuple(w for w, t in self.words)

        assert self.caller in ('A', 'B')

    def damsl_act_tag(self):
        return self._damsl_act_tag

    def __str__(self):
        return '{s.caller} {s._damsl_act_tag} {s.text}'.format(s=self)

    def utterance_tokens(self, ngram_len=1):
        assert ngram_len == 1
        return self.words

    def append_text(self, other):
        self.words = self.words + other.words


def reconnect_divided_utterances(swda, *, lemmatize, pos):
    """Get rid of the '+' tag and stick them to the first utterance."""
    utterances = swda.iter_utterances(display_progress=False)

    result = []

    for utterance in utterances:
        u = U(utterance, lemmatize=lemmatize, pos=pos)

        if u.damsl_act_tag() == '+':
            for p_utterance in reversed(result):
                if u.conversation_no != p_utterance.conversation_no:
                    logger.warning(
                        'There is no utterance before an utterance tagged with + in conversation %d.',
                        u.conversation_no,
                    )
                    break

                if u.caller == p_utterance.caller:
                    p_utterance.append_text(u)
                    break
        else:
            result.append(u)

    return result


class Dispatcher(dispatcher.Dispatcher):
    global__lemmatize = '', False, 'Lemmatize the utterance words before retrieving their vectors.'
    global__n_folds = 'f', 3, 'The number of folds used for cross validation.'
    global__space = '', 'space.h5', 'The space file.'
    global__swda = '', './swda', 'The path to the Switchboard corpus.'
    global__test_split = '', 'downloads/switchboard/ws97-test-convs.list.txt', 'The testing splits'
    global__train_split = '', 'downloads/switchboard/ws97-train-convs.list.txt', 'The training splits'
    global__pos = '', False, 'Use word, POS pairs.'

    @dispatcher.Resource
    def swda(self):
        return CorpusReader(self.kwargs['swda'])

    @dispatcher.Resource
    def utterances(self):
        logger.info('Reading the utterances.')

        return reconnect_divided_utterances(self.swda, lemmatize=self.lemmatize, pos=self.pos)

    @dispatcher.Resource
    def train_split(self):
        return get_conversation_ids(self.kwargs['train_split'])

    @dispatcher.Resource
    def test_split(self):
        return get_conversation_ids(self.kwargs['test_split'])

    @dispatcher.Resource
    def train_utterances(self):
        train_utterances = [u for u in self.utterances if u.conversation_no in self.train_split]

        if self.limit:
            train_utterances = [u for u in self.utterances if u.conversation_no in self.train_split][:self.limit]

        assert train_utterances
        return train_utterances

    @dispatcher.Resource
    def test_utterances(self):
        test_utterances = [u for u in self.utterances if u.conversation_no in self.test_split]

        assert test_utterances
        return test_utterances

    @dispatcher.Resource
    def space(self):
        return models.read_space_from_file(self.kwargs['space'])

    @dispatcher.Resource
    def y_train(self):
        return [u.damsl_act_tag() for u in self.train_utterances]

    @dispatcher.Resource
    def y_test(self):
        return [u.damsl_act_tag() for u in self.test_utterances]


dispatcher = Dispatcher()
command = dispatcher.command


def document_word(args):
    i, u, dictionary, ngram_len = args
    js = dictionary.loc[u.utterance_tokens(ngram_len=ngram_len)].id

    result = []
    for j in js:
        if np.isfinite(j):
            result.append([i, j, 1])

    if not result:
        result = [[i, 0, 0]]

    return result


@command()
def tokens(
    train_utterances,
    pos,
):
    words = chain.from_iterable(u.utterance_tokens(ngram_len=1) for u in train_utterances)

    freq = Counter(words)

    for w, f in freq.most_common():
        if pos:
            w = ','.join(w)
        print(w, '\t', f)


@command()
def plain_lsa(
    train_utterances,
    test_utterances,
    pool,
    y_train,
    y_test,
    templates_env,
    n_folds,
    n_jobs,
    ngram_len=('', 1, 'Length of the tokens (bigrams, ngrams).'),
):
    """Perform the Plain LSA method."""

    words = list(set(chain.from_iterable(u.utterance_tokens(ngram_len=ngram_len) for u in train_utterances)))
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

    evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_folds=n_folds,
        n_jobs=n_jobs,
        templates_env=templates_env,
        store_metadata={},
        paper='Serafin et al. 2003',
        pool=pool,
    )


def space_compose(args):
    u, composer = args
    return composer(*u.utterance_tokens())


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
    word_composition_operator=('', 'add', 'What operator use for composition. [add|mult]'),
    concatinate_prev_utterace=('', False, 'Concatenate the vector of a current utterance with the vector of the precious utterance.'),
):

    if word_composition_operator == 'mult':
        composer = space.multiply
    else:
        composer = space.add

    def extract_features(utterances):

        logger.info('Extracting features.')
        # This might be inefficient, because the space object is passed to the pool.
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

            logger.debug('Hstacking utterances with their previous utterances.')
            X = hstack([X, prev_X], format='csr')

        return X

    X_train = extract_features(train_utterances)
    X_test = extract_features(test_utterances)

    evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_folds=n_folds,
        n_jobs=n_jobs,
        templates_env=templates_env,
        store_metadata={},
        paper='Comp sem.',
        pool=pool,
    )


def evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    templates_env,
    store_metadata,
    n_folds,
    n_jobs,
    paper,
    pool,
):

    pipeline = Pipeline(
        [
            ('svd', TruncatedSVD(n_components=50)),
            ('nn', KNeighborsClassifier()),
        ]
    )

    logger.info('Training.')
    pipeline.fit(X_train, y_train)

    logger.info('Predicting %d labels.', X_test.shape[0])
    y_predicted = pipeline.predict(X_test)

    prfs = precision_recall_fscore_support(y_test, y_predicted)
    util.display(
        templates_env.get_template('classification_report.rst').render(
            argv=' '.join(sys.argv) if not util.inside_ipython() else 'ipython',
            paper=paper,
            clf=pipeline,
            tprfs=zip(unique_labels(y_test, y_predicted), *prfs),
            p_avg=np.average(prfs[0], weights=prfs[3]),
            r_avg=np.average(prfs[1], weights=prfs[3]),
            f_avg=np.average(prfs[2], weights=prfs[3]),
            s_sum=np.sum(prfs[3]),
            store_metadata=store_metadata,
            accuracy=accuracy_score(y_test, y_predicted),
        )
    )

    pd.DataFrame(y_predicted).to_csv('out.csv')
    pd.DataFrame(y_test).to_csv('y_test.csv')
