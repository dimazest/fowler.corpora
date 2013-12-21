import csv
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def load_cooccurrence(args):
    file_name, targets, context = args

    logger.info('Processing %s', file_name)

    frame = pd.read_csv(
        str(file_name),
        names=('ngram', 'context', 'count'),
        encoding='utf8',
        compression='gzip',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )

    piece = (
        frame
        .merge(targets, left_on='ngram', right_index=True, sort=False)
        .merge(context, left_on='context', right_index=True, sort=False, suffixes=('_target', '_context'))
        [['id_target', 'id_context', 'count']]
    )

    piece = piece.groupby(['id_target', 'id_context'], as_index=False).sum()

    return piece
