"""The Google Books Ngram Viewer dataset helper routines."""
import csv

from opster import Dispatcher
from py.path import local

import numpy as np
import pandas as pd

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def dictionary(
    input_dir=('i', local('./downloads/google_ngrams/1'), 'The path to the directory with the Google unigram files.'),
    output=('o', 'dictionary.csv', 'The output file.'),
    with_pos=('', False, 'Include ngrams that are POS tagged.'),
):
    """Build the dictionary, sorted by frequency.

    The items in the are sorted by frequency. The output contains two columns
    separated by tab. The first column is the element, the second is its frequency.

    """

    pieces = []
    for file_name in input_dir.listdir():
        print('Processing {}'.format(file_name))
        frame = pd.read_csv(
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
        )

        frame['ngram'].fillna('U+F8F0:<INVALIDCHARACTER>', inplace=True)

        if not with_pos:
            frame = frame[np.invert(frame['ngram'].str.contains('_'))]

        pieces.append(frame.groupby('ngram', as_index=False).sum())

    counts = pd.concat(pieces, ignore_index=True)
    counts.sort(
        'count',
        inplace=True,
        ascending=False,
    )

    counts.to_csv(
        output,
        header=False,
        sep='\t',
        index=False,
    )


@command()
def cooccurrence(
    dictionary=('d', 'dictionary.csv.gz', 'The dictionary file.'),
    input_dir=('i', local('./downloads/google_ngrams/5_cooccurrence'), 'The path to the directory with the Google unigram files.'),
    with_pos=('', False, 'Include ngrams that are POS tagged.'),
    output=('o', 'cooccurrence.csv', 'The output file.'),
):
    """Build the cooccurrence matrix.

    :param str dictionary: the file with contexts that

    """
    dictionary = pd.read_csv(
        dictionary,
        names=('ngram', 'count'),
        usecols=('ngram', ),
        index_col='ngram',
        encoding='utf8',
        compression='gzip',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )
    dictionary['id'] = pd.Series(np.arange(len(dictionary)), index=dictionary.index)

    pieces = []
    for file_name in input_dir.listdir(sort=True):
        print('Processing {}'.format(file_name))

        frame = pd.read_csv(
            str(file_name),
            names=('ngram', 'context', 'count'),
            encoding='utf8',
            compression='gzip',
            delim_whitespace=True,
            quoting=csv.QUOTE_NONE,
        )

        frame['ngram'].fillna('U+F8F0:<INVALIDCHARACTER>', inplace=True)

        if not with_pos:
            frame = frame[np.invert(frame['ngram'].str.contains('_'))]

        piece = (
            frame
            .merge(dictionary, left_on='ngram', right_index=True, sort=False)
            .merge(dictionary, left_on='context', right_index=True, sort=False, suffixes=('_target', '_context'))
            [['id_target', 'id_context', 'count']]
        )

        piece = piece.groupby(['id_target', 'id_context'], as_index=False).sum()
        pieces.append(piece)

    cooc = pd.concat(pieces, ignore_index=True).groupby(['id_target', 'id_context']).sum()

    cooc.to_csv(
        output,
        header=False,
        sep='\t',
    )
