import logging
from collections import Counter, deque
from itertools import islice, chain


logger = logging.getLogger(__name__)


def count_cooccurrence(words, window_size=5):
    """Count word couccurrence.

    :param iter words: the sequence of words.
    :param int window_size: the symmetric window size.

    # :return: a Counter of the cooccurrece pairs.

    """
    words = iter(words)

    before = deque(islice(words, window_size))
    after = deque([], maxlen=window_size)

    counts = {}

    while before:
        try:
            word = next(words)
        except StopIteration:
            '''There are no words anymore.'''
        else:
            before.append(word)

        target = before.popleft()

        counts.setdefault(target, Counter()).update(chain(before, after))

        after.append(target)

    pairs = chain.from_iterable(((t, c, n) for c, n in cs.items()) for t, cs in counts.items())
    return list(pairs)
