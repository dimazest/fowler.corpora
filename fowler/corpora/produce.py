"""Produce helpers."""
import logging

import pandas as pd
from nltk.corpus import stopwords

from fowler.corpora.dispatcher import Dispatcher


class ProduceDispatcher(Dispatcher):
    global__target = '', '', 'Output filename.'

logger = logging.getLogger(__name__)
dispatcher = ProduceDispatcher()
command = dispatcher.command


@command()
def filter_stopwords(
    target,
    dictionary=('', '', 'Dictionary filename.'),
):

    df = pd.read_hdf(dictionary, key='dictionary')
    df['rank'] = list(range(len(df)))
    df = df.set_index('ngram')

    df = df.loc[stopwords.words('english')]
    df = df.sort('rank')
    df.to_csv(target, encoding='utf-8')


@command()
def filter_verbs(
    target,
    targets=('', '', 'Targets filename.'),
):
    targets = pd.read_csv(targets, encoding='utf-8')
    verbs = targets[targets['tag'] == 'V']
    verbs.to_csv(
        target,
        encoding='utf-8',
        index=False,
        columns=['ngram'],
        header=False,
    )


@command()
def select_targets(
    target,
    dataset_dictionary=('', '', 'Dataset dictionary filename.'),

):
    target_dictionary = pd.read_hdf(dataset_dictionary, 'dictionary')
    assert target_dictionary.set_index(['ngram', 'tag']).index.is_unique

    target_dictionary[['ngram', 'tag']].to_csv(target, index=False, encoding='utf-8')


@command()
def select_context(
    target,
    corpus_dictionary=('', '', 'Dictionary filename.'),
    corpus_type=('', '', 'Corpus arg'),
    size=('', 2000, 'Size arg'),
):
    corpus_dictionary = pd.read_hdf(corpus_dictionary, 'dictionary')
    assert corpus_dictionary.set_index(['ngram', 'tag']).index.is_unique

    if corpus_type == 'bnc':
        nvaa = corpus_dictionary[corpus_dictionary['tag'].isin(['SUBST', 'VERB', 'ADJ', 'ADV'])]
    else:
        nvaa = corpus_dictionary[corpus_dictionary['tag'].isin(['N', 'V', 'J', 'R'])]

    head = nvaa.head(size)[['ngram', 'tag']]

    assert head.set_index(['ngram', 'tag']).index.is_unique

    head.to_csv(target, index=False, encoding='utf-8')


@command()
def all_features(
    target,
    dictionary=('', '', 'Dictionary filename.'),
    limit=('', 5, 'The minimal number of times a feature has to occur to be counted.')
):
    dictionary = pd.read_hdf(dictionary, 'dictionary')
    dictionary.drop(dictionary[dictionary['count'] < limit].index, inplace=True)
    dictionary.to_csv(
        target,
        columns=('ngram', 'tag'),
        encoding='utf-8',
        index=False,
    )


@command()
def convert_results(
    target,
    experiment=('', '', 'An .h5 experiment results file.'),
):
    pd.read_hdf(experiment, key='dataset').to_csv(target)
