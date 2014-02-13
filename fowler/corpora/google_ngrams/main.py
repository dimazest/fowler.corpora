"""The Google Books Ngram Viewer dataset helper routines."""
import logging

from py.path import local

import pandas as pd

from fowler.corpora import dispatcher

from fowler.corpora.space.util import write_space

from .util import load_cooccurrence, load_dictionary


logger = logging.getLogger(__name__)


class Dispatcher(dispatcher.Dispatcher, dispatcher.SpaceCreationMixin):
    """A concrete dispatcher."""


dispatcher = Dispatcher()
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
    """Build the co-occurrence matrix."""
    file_names = input_dir.listdir(sort=True)
    pieces = pool.map(load_cooccurrence, ((f, targets, context) for f in file_names))

    # Get rid of empty frames
    pieces = list(filter(len, pieces))

    while len(pieces) > 1:
        logger.info('Pairs left %s', len(pieces))

        if divmod(len(pieces), 2)[1]:
            odd = [pieces.pop()]
        else:
            odd = []

        pieces = list(pool.map(group_sum, get_pairs(pieces))) + odd

    matrix, = pieces

    write_space(output, context, targets, matrix)


def group_sum(args):
    f1, f2 = args
    return f1.append(f2).groupby(level=['id_target', 'id_context']).sum()


def get_pairs(seq):
    assert not divmod(len(seq), 2)[1]
    seq = iter(seq)

    while True:
        f = next(seq)
        s = next(seq)
        yield f, s
