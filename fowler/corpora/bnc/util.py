import logging
from collections import deque
from itertools import islice, chain


logger = logging.getLogger(__name__)


def co_occurrences(words, window_size_before, window_size_after):
    """Yield word co-occurrence.

    :param iter words: the sequence of words.
    :param int window_size_before: window size before the target token.
    :param int window_size_after: window size after the target token.

    """
    words = iter(words)

    target = next(words)
    before = deque([], maxlen=window_size_before)
    after = deque(islice(words, window_size_after))

    while True:
        for context in chain(before, after):
            yield target + context

        before.append(target)

        try:
            after.append(next(words))
        except StopIteration:
            pass

        try:
            target = after.popleft()
        except IndexError:
            break
