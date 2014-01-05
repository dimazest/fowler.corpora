"""Dictionary helpers."""
import pandas as pd

from fowler.corpora.dispatcher import Dispatcher


def middleware_hook(kwargs, f_args):
    if 'dictionary' in f_args:
        kwargs['dictionary'] = pd.read_hdf(
            kwargs['dictionary'],
            key=kwargs['input_key'],
        )

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
def limit(
    dictionary,
    limit=('l', 30000, 'Number of rows to leave in the index.'),
    output=('o', 'dicitionary_limited.h5', 'The output file.'),
    output_key=('', 'dictionary', 'An identifier for the group in the store.'),
):
    dictionary = dictionary.head(limit)
    dictionary.to_hdf(
        output,
        output_key,
        mode='w',
        complevel=9,
        complib='zlib',
    )


@command()
def select(
    dictionary,
    output=('o', 'ngrams.csv', 'The output file.'),
    slice_start=('', 0, 'Number of ngrams in the output.'),
    slice_end=('', 0, 'Number of ngrams in the output.'),
    pos_tagged=('', False, 'Get POS tagged items.'),
):
    """Retrieve ngrams from the dictionary.

    Tokens in with parts of speech should contain an underscore. For example,
    ``the_DET``.

    """
    del dictionary['count']

    if pos_tagged:
        ngrams = dictionary[dictionary['ngram'].str.contains('^[^_]+_')]
    else:
        ngrams = dictionary[~dictionary['ngram'].str.contains('_')]

    if slice_start or slice_end:
        ngrams = ngrams[slice_start or None:slice_end or None]

    print(ngrams)

    ngrams.to_csv(
        output,
        sep='\t',
        header=False,
        index=False,
    )

