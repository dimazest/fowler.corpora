import logging
from collections import deque
from itertools import islice, chain


logger = logging.getLogger(__name__)


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
        yield target, tuple(chain(before, after))

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
