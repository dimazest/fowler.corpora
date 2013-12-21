"""The Google Books Ngram Viewer dataset helper routines."""
import csv
from multiprocessing import Pool

from py.path import local

import numpy as np
import pandas as pd

from fowler.corpora.main.options import Dispatcher

from .util import load_cooccurrence


def middleware_hook(kwargs, f_args):
    if 'pool' in f_args:
        kwargs['pool'] = Pool(kwargs['jobs_num'] or None)


dispatcher = Dispatcher(middleware_hook=middleware_hook)
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
    pool=None,
    context=('c', 'context.csv.gz', 'The file with context words.'),
    targets=('t', 'targets.csv.gz', 'The file with target words.'),
    input_dir=(
        'i',
        local('./downloads/google_ngrams/5_cooccurrence'),
        'The path to the directory with the co-occurence.',
    ),
    output=('o', 'matrix.h5', 'The output matrix file.'),
):

    """Build the cooccurrence matrix."""
    context = pd.read_csv(
        context,
        names=('ngram', 'count'),
        usecols=('ngram', ),
        index_col='ngram',
        encoding='utf8',
        compression='gzip',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )
    context['id'] = pd.Series(np.arange(len(context)), index=context.index)

    targets = pd.read_csv(
        targets,
        names=('ngram', ),
        index_col='ngram',
        encoding='utf8',
        compression='gzip',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )
    targets['id'] = pd.Series(np.arange(len(targets)), index=targets.index)

    file_names = input_dir.listdir(sort=True)
    pieces = pool.map(load_cooccurrence, ((f, targets, context) for f in file_names))
    # for file_name in
    #     pieces.append(load_cooccurrence(file_name, targets, context))
    matrix = pd.concat(pieces, ignore_index=True).groupby(['id_target', 'id_context']).sum()

    with pd.get_store(output, mode='w') as store:

        store['context'] = context
        store['targets'] = targets
        store['matrix'] = matrix

        # TODO it would be nice to write metadata, e.g. the command the file
        # was generated, the date and so on.
