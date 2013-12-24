"""The Google Books Ngram Viewer dataset helper routines."""
import csv
from multiprocessing import Pool

from py.path import local

import numpy as np
import pandas as pd

from fowler.corpora.dispatcher import Dispatcher

from .util import load_cooccurrence, load_dictionary


def middleware_hook(kwargs, f_args):
    if 'pool' in f_args:
        kwargs['pool'] = Pool(kwargs['jobs_num'] or None)


dispatcher = Dispatcher(middleware_hook=middleware_hook)
command = dispatcher.command


@command()
def dictionary(
    pool,
    input_dir=('i', local('./downloads/google_ngrams/1'), 'The path to the directory with the Google unigram files.'),
    output=('o', 'dictionary.h5', 'The output file.'),
    output_key=('', 'dictionary', 'An identifier for the group in the store.')
):
    """Build the dictionary, sorted by frequency.

    The items in the are sorted by frequency. The output contains two columns
    separated by tab. The first column is the element, the second is its frequency.

    """

    file_names = sorted(input_dir.listdir())
    pieces = pool.map(load_dictionary, file_names)
    counts = pd.concat(pieces, ignore_index=True)
    counts.sort(
        'count',
        inplace=True,
        ascending=False,
    )
    counts.reset_index(drop=True, inplace=True)

    print(counts)

    counts.to_hdf(
        output,
        key=output_key,
        mode='w',
        complevel=9,
        complib='zlib',
    )


@command()
def cooccurrence(
    pool=None,
    context=('c', 'context.csv', 'The file with context words.'),
    targets=('t', 'targets.csv', 'The file with target words.'),
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
        names=('id', 'ngram'),
        index_col='ngram',
        encoding='utf8',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )

    targets = pd.read_csv(
        targets,
        names=('ngram', ),
        index_col='ngram',
        encoding='utf8',
        delim_whitespace=True,
        quoting=csv.QUOTE_NONE,
    )
    targets['id'] = pd.Series(np.arange(len(targets)), index=targets.index)

    file_names = input_dir.listdir(sort=True)
    pieces = pool.map(load_cooccurrence, ((f, targets, context) for f in file_names))
    matrix = pd.concat(pieces, ignore_index=True).groupby(['id_target', 'id_context']).sum()

    with pd.get_store(output, mode='w', complevel=9, complib='zlib',) as store:

        store['context'] = context
        store['targets'] = targets
        store['matrix'] = matrix

        # TODO it would be nice to write metadata, e.g. the command the file
        # was generated, the date and so on.
