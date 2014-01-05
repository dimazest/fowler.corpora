from collections import deque, Counter
from itertools import chain

import gzip

import pandas as pd

from fowler.corpora.dispatcher import Dispatcher

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def cooccurrence(
    window_size=(
        'w',
        5,
        "The size of the simmetric window. How many items to include to the window from token's each side.",
    ),
    path=('p', 'some_text.txt', 'The path to the input file.'),
    output=('o', 'coocurrence.csv.gz', 'The cooccurrence file'),
):
    """Get word co-ccurrence frequncies."""
    with open(path) as f:
        tokens = _tokens(f)
        ngrams = _ngrams(window_size, tokens)
        cooc = Counter(_cooc(ngrams, window_size))

    with gzip.open(output, 'wt') as f:
        for (t, c), freq in cooc.most_common():
            f.write('{}\t{}\t{}\n'.format(t, c, freq))


def _tokens(lines):
    return chain.from_iterable(l.split() for l in lines)


def _ngrams(window_size, tokens):
    ngram_len = window_size + 1 + window_size
    ngram = deque(['<BEGIN>'] * ngram_len, ngram_len)

    tokens = chain(tokens, ['<END>'] * ngram_len)

    for token in tokens:
        ngram.append(token)

        yield tuple(ngram)


def _cooc(ngrams, window_size):
    for ngram in ngrams:
        target_index = window_size + 1
        target = ngram[target_index]
        context = ngram[:target_index] + ngram[target_index + 1:]

        for c in context:
            yield target, c


@command()
def dictionary(
    path=('p', 'some_text.txt', 'The path to the input file.'),
    output=('o', 'dictionary.h5', 'The dictionary file'),
    output_key=('', 'dictionary', 'An identifier for the group in the store.')
):
    with open(path) as f:
        tokens = _tokens(f)
        token_freq = Counter(tokens)

        counts = pd.DataFrame(
            token_freq.most_common(),
            columns=['ngram', 'count'],
        )

    counts.to_hdf(
        output,
        key=output_key,
        mode='w',
        complevel=9,
        complib='zlib',
    )


