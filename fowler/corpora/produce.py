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
def features(
    target,
    dictionary=('', '', 'Dictionary filename.'),
    mintf=('', 1000, 'The minimal number of times a feature has to occur to be counted.')
):
    dictionary = pd.read_hdf(dictionary, 'dictionary')

    dictionary.drop(dictionary[dictionary['count'] < mintf].index, inplace=True)
    dictionary.to_csv(
        target,
        columns=('ngram', 'tag'),
        encoding='utf-8',
        index=False,
    )


@command()
def combine(
    target,
    dataset_targets=('', '', ''),
    corpus_contexts=('', '', ''),
):
    dataset_targets = pd.read_csv(dataset_targets, encoding='utf-8')
    corpus_contexts = pd.read_csv(corpus_contexts, encoding='utf-8')

    pd.concat(
        [corpus_contexts, dataset_targets]
    ).drop_duplicates().to_csv(
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
