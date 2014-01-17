import csv
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def load_cooccurrence(args):
    """Retriece the cooccurrances that contain both passed targets and contexts."""
    file_name, targets, context = args

    logger.info('Processing %s', file_name)

    frame = pd.read_csv(
        str(file_name),
        names=('ngram', 'context', 'count'),
        encoding='utf8',
        compression='gzip',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
        dtype={
            'ngram': str,
            'context': str,
            'count': 'uint64',
        }
    )

    piece = (
        frame
        .merge(targets, left_on='ngram', right_index=True, sort=False)
        .merge(context, left_on='context', right_index=True, sort=False, suffixes=('_target', '_context'))
        [['id_target', 'id_context', 'count']]
    )

    piece = piece.groupby(
        ['id_target', 'id_context'],
    ).sum()

    return piece


def load_dictionary(file_name):
    logger.info('Processing %s', file_name)

    return pd.read_csv(
        str(file_name),
        names=('ngram', 'year', 'count', 'volume_count'),
        usecols=('ngram', 'count'),
        header=0,
        encoding='utf8',
        compression='gzip',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
        dtype={
            'ngram': str,
            'count': int,
        }
    ).groupby('ngram', as_index=False).sum()
