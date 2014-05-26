import logging

from collections import deque
from itertools import islice, chain

import pandas as pd


logger = logging.getLogger(__name__)


def count_cooccurrence(words, window_size=5):
    """Count word co-occurrence.

    :param iter words: the sequence of words.
    :param int window_size: the symmetric window size.
    :param bool pos_tag: use POS tags.

    :return: a pandas.DataFrame of the co-occurrence counts.
        The key columns are::

            'target', 'target_tag', 'context', 'context_tag'

        The only content column is ``count``, that specifies how many times a
        target and a context co-occurred.

    """
    words = iter(words)

    before = deque(islice(words, window_size))
    after = deque([], maxlen=window_size)

    def co_occurrences():
        while before:
            try:
                word = next(words)
            except StopIteration:
                '''There are no words anymore.'''
            else:
                before.append(word)

            target = before.popleft()

            for context in chain(before, after):
                yield target + context

            after.append(target)

    columns = 'target', 'target_tag', 'context', 'context_tag'

    counts = pd.DataFrame(
        co_occurrences(),
        columns=columns,
    )
    counts['count'] = 1

    return counts.groupby(columns, as_index=False).sum()
