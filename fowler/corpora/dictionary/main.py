"""Dictionary helpers."""
from fowler.corpora.dispatcher import Dispatcher, DictionaryMixin


class DictionaryDispatcher(Dispatcher, DictionaryMixin):
    """Dictionary dispatcher."""

dispatcher = DictionaryDispatcher()
command = dispatcher.command


@command()
def limit(
    dictionary,
    dictionary_key,
    limit=('l', 30000, 'Number of rows to leave in the index.'),
    output=('o', 'dicitionary_limited.h5', 'The output file.'),
):
    dictionary = dictionary.head(limit)
    dictionary.to_hdf(
        output,
        dictionary_key,
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

