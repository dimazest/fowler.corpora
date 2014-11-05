import logging
from collections import deque
from itertools import islice, chain

import pandas as pd


logger = logging.getLogger(__name__)


def count_cooccurrence(words, window_size=5):
    """Count word co-occurrence.

    :param iter words: the sequence of words.
    :param window_size: if a scalar: the symmetric window size,
                        if a tuple: the assymetric window size.

    :return: a pandas.DataFrame of the co-occurrence counts.
        The key columns are::

            'target', 'target_tag', 'context', 'context_tag'

        The only content column is ``count``, that specifies how many times a
        target and a context co-occurred.

    """

    columns = 'target', 'target_tag', 'context', 'context_tag'

    counts = pd.DataFrame(
        co_occurrences(words, window_size),
        columns=columns,
    )
    counts['count'] = 1

    return counts.groupby(columns, as_index=False).sum()


def co_occurrences(words, window_size=5):
    """Yield word co-occurrence.

    :param iter words: the sequence of words.
    :param window_size: if a scalar: the symmetric window size,
                        if a tuple: the assymetric window size.

    """
    words = iter(words)

    try:
        window_size_before, window_size_after = window_size
    except TypeError:
        window_size_before = window_size_after = window_size

    target = next(words)
    before = deque([], maxlen=window_size_before)
    after = deque(islice(words, window_size_after))

    while True:
        for context in chain(before, after):
            yield target + context

        before.append(target)

        try:
            word = next(words)
        except StopIteration:
            '''There are no more words.'''
            if not after:
                break
        else:
            after.append(word)

        target = after.popleft()
