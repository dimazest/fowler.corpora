import logging

from collections import deque, namedtuple
from itertools import islice, chain, takewhile, filterfalse

import pandas as pd


logger = logging.getLogger(__name__)

CCGToken = namedtuple('CCGToken', 'word, stem, tag')


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


def ccg_bnc_iter(f_name, postprocessor):
    logger.debug('Processing %s', f_name)

    with open(f_name, 'rt', encoding='utf8') as f:
        # Get rid of trailing whitespace.
        lines = (l.strip() for l in f)

        while True:
            # Sentences are split by an empty line.
            sentence = list(takewhile(bool, lines))

            if not sentence:
                # No line were taken, this means all the file has be read!
                break

            # Take extra care of comments.
            sentence = list(filterfalse(lambda l: l.startswith('#'), sentence))

            if not sentence:
                # If we got nothing, but comments: skip.
                continue

            *dependencies, c = sentence
            tokens = dict(parse_tokens(c))

            yield from postprocessor(dependencies, tokens)


def parse_tokens(c):
    """Parse and retrieve token position, word, stem and tag from a C&C parse."""
    assert c[:4] == '<c> '
    c = c[4:]

    for position, token in enumerate(c.split()):
        word, stem, tag, *_ = token.split('|')

        yield position, CCGToken(word, stem, tag)
