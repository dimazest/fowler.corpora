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
def select_entities(
    target,
    dictionary=('', '', 'Dictionary filename.'),
    corpus_arg=('', '', 'Corpus arg'),
    entity_arg=('', '', 'Entity arg'),
    size_arg=('', 2000, 'Entity arg'),
    verb_arguments=('', '', 'Entity arg'),
    targets=('', '', 'Targets filename.'),
    cutoff_size=('', 0, 'Targets filename.')
):
    dictionary = pd.read_hdf(dictionary, 'dictionary')
    assert dictionary.set_index(['ngram', 'tag']).index.is_unique

    if corpus_arg == 'bnc':
        nvaa = dictionary[dictionary['tag'].isin(['SUBST', 'VERB', 'ADJ', 'ADV'])]
    else:
        nvaa = dictionary[dictionary['tag'].isin(['N', 'V', 'J', 'R'])]

    if entity_arg == 'context':
        head = nvaa.head(size_arg)[['ngram', 'tag']]

    elif entity_arg == 'targets':
        dictionary.set_index(['ngram', 'tag'], inplace=True)

        dataset = pd.read_csv(targets, encoding='utf-8')
        if verb_arguments:
            verb_arguments = pd.read_csv(verb_arguments, encoding='utf-8').dropna()

            verb_argument_counts = dictionary.loc[[tuple(x) for x in verb_arguments.values]]
            verb_argument_counts = verb_argument_counts[verb_argument_counts['count'] >= cutoff_size]

            verb_arguments = verb_argument_counts.reset_index()[['ngram', 'tag']]

            head = pd.concat([dataset, verb_arguments]).drop_duplicates()
        else:
            head = dataset.drop_duplicates()

        if size_arg:
            head = pd.concat([head, nvaa.head(size_arg)[['ngram', 'tag']]]).drop_duplicates()

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
