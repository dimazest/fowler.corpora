import logging

from collections import deque, namedtuple
from itertools import islice, chain, filterfalse, takewhile, groupby

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


def collect_verb_subject_object(args):
    """Retrieve verb together with it's subject and object from a C&C parsed file.

    File format description [1].

    [1] http://svn.ask.it.usyd.edu.au/trac/candc/wiki/MarkedUp

    """
    f_name, tag_first_letter = args
    columns = 'verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag'

    result = list(ccg_bnc_iter(f_name, _collect_verb_subject_object, tag_first_letter=tag_first_letter))

    if result:
        result = pd.DataFrame(
            result,
            columns=columns,
        )
        result['count'] = 1

        return result.groupby(columns, as_index=False).sum()


def _collect_verb_subject_object(dependencies, tokens):
    dependencies = sorted(parse_dependencies(dependencies))

    for head_id, group in groupby(dependencies, lambda r: r[0]):
        group = list(group)

        try:
            (_, obj, obj_id), (_, subj, subj_id) = sorted(g for g in group if g[1] in ('dobj', 'ncsubj'))
        except ValueError:
            pass
        else:
            if obj == 'dobj'and subj == 'ncsubj':

                try:
                    yield tuple(chain(tokens[head_id], tokens[subj_id], tokens[obj_id]))
                except KeyError:
                    logger.debug('Invalid group %s', group)


def ccg_words(args):
    f_name, tag_first_letter = args
    result = list(ccg_bnc_iter(f_name, _ccg_words, tag_first_letter=True))

    columns = 'ngram', 'stem', 'tag'
    if result:
        result = pd.DataFrame(
            result,
            columns=columns,
        )
        result['count'] = 1

        return result.groupby(columns, as_index=False).sum()


def _ccg_words(dependencies, tokens):
    yield from tokens.values()


def ccg_bnc_iter(f_name, postprocessor, tag_first_letter=False):
    logger.debug('Processing %s', f_name)

    with open(f_name, 'rt', encoding='utf8') as f:
        # Get rid of trailing whitespace.
        lines = (l.strip() for l in f)

        while True:
            # Sentences are split by an empty line.
            sentence = list(takewhile(bool, lines))

            if not sentence:
                # No line were taken, this means all the file has be read!f.
                break

            # Take extra care of comments.
            sentence = list(filterfalse(lambda l: l.startswith('#'), sentence))

            if not sentence:
                # If we got nothing, but comments: skip.
                continue

            *dependencies, c = sentence
            tokens = dict(parse_tokens(c, tag_first_letter=tag_first_letter))

            yield from postprocessor(dependencies, tokens)


def parse_dependencies(dependencies):
    """Parse and filter out verb subject/object dependencies from a C&C parse."""
    for dependency in dependencies:
        assert dependency[0] == '('
        assert dependency[-1] == ')'
        dependency = dependency[1:-1]

        try:
            relation, head, dependant, *_  = dependency.split()
        except ValueError:
            logger.debug('Invalid dependency: %s', dependency)
            break

        if relation in set(['ncsubj', 'dobj']):
            yield (
                int(head.split('_')[1]),
                relation,
                int(dependant.split('_')[1]),
            )


def parse_tokens(c, tag_first_letter=False):
    """Parse and retrieve token position, word, stem and tag from a C&C parse.

    Replaces C&C tags with the BNC pos tags.

    """
    assert c[:4] == '<c> '
    c = c[4:]

    for position, token in enumerate(c.split()):
        word, stem, tag, *_ = token.split('|')

        if tag_first_letter:
            tag = tag[0]

        yield position, CCGToken(word, stem, tag)
