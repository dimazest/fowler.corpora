"""Dictionary helpers."""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fowler.corpora.dispatcher import Dispatcher


def middleware_hook(kwargs, f_args):
    if 'dictionary' in f_args:
        kwargs['dictionary'] = pd.read_hdf(
            kwargs['dictionary'],
            key=kwargs['input_key'],
        )
    if 'pool' in f_args:
        kwargs['pool'] = Pool(kwargs['jobs_num'] or None)

    if 'input_key' not in f_args:
        del kwargs['input_key']

dispatcher = Dispatcher(
    middleware_hook=middleware_hook,
    globaloptions=(
        ('d', 'dictionary', 'dictionary.h5', 'The input dictionary.'),
        ('', 'input_key', 'dictionary', 'An identifier for the group in the store.')
    ),
)
command = dispatcher.command


@command()
def filter(
    dictionary,
    output=('o', 'ngrams.h5', 'The output file.'),
    output_key=('', 'ngrams', 'An identifier for the group in the store.'),
    limit=('l', 0, 'Number of ngrams in the output.'),
):
    """Retrieve ngrams from the dictionary.

    Tokens in with parts of speech should contain an underscore. For example,
    ``the_DET``.

    """

    dictionary = dictionary[dictionary['ngram'].str.contains('_')]
    dictionary.reset_index(drop=True, inplace=True)

    ngrams = dictionary['ngram']

    if limit:
        ngrams = ngrams.head(limit)

    ngrams = pd.DataFrame(ngrams.index, index=ngrams, columns=('id', ))
    print(ngrams)

    ngrams.to_hdf(
        output,
        output_key,
        mode='w',
        complevel=9,
        complib='zlib',
    )

