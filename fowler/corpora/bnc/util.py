import logging
from collections import Counter, deque
from itertools import islice, chain, filterfalse, takewhile, groupby


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


def parse_tokens(c):
    """Parse and retrieve token position, word, stem and tag from a C&C parse.

    Replaces C&C tags with the BNC pos tags.

    """
    assert c[:4] == '<c> '
    c = c[4:]

    c5tags = {
        '$': '__$__',  # c
        ',': 'PUN',  # ,
        '.': 'PUN',  # .
        ':': 'PUN',  # :
        ';': 'PUN',  # ;
        'AS': '__AS__',  # as
        'CC': 'CONJ',  # and
        'CD': 'ADJ',  # three
        'DT': 'ART',  # the
        'EX': 'PRON',  # there
        'FW': 'SUBJ',  # nightingale
        'IN': 'PREP',  # of
        'JJ': 'ADJ',  # delayed
        'JJR': 'ADJ',  # greater
        'JJS': 'ADJ',  # biggest
        'LQU': '__LQU__',  # ``
        'LRB': 'PUN',  # (
        'LS': '__LS__',  # b
        'MD': 'VERB',  # could
        'NN': 'SUBST',  # tax
        'NNP': 'SUBST',  # London
        'NNPS': 'SUBST',  # allowances
        'NNS': 'SUBST',  # payment
        'NP': 'SUBST',  # neither
        'PDT': 'ADJ',  # all
        'POS': 'UNC',  # 's
        'PRP$': 'PRON',  # its, their
        'PRP': 'PRON',  # it
        'RB': 'ADV',  # n't, soon
        'RBR': 'ADV',  # more
        'RBS': 'ADV',  # most
        'RP': 'ADV',  # up
        'RQU': '__RQU__',  # ''
        'RRB': 'PUN',  # )
        'SO': 'CONJ',  # so
        'SYM': '__SYM__',
        'TO': '__TO__',  # to
        'UH': '__UH__',  # oh
        'VB': 'VERB',  # pay
        'VBD': 'VERB',  # commend
        'VBG': 'VERB',  # operate
        'VBN': 'VERB',  # be
        'VBP': 'VERB',  # mount
        'VBZ': 'VERB',  # have
        'WDT': 'PRON',  # which
        'WP$': 'PRON',  # whose
        'WP': 'PRON',  # who
        'WRB': 'ADV',  # where
    }

    for position, token in enumerate(c.split()):
        word, stem, tag, *_ = token.split('|')

        yield position, (word, stem, c5tags[tag])


def collect_verb_subject_object(f_name):
    """Retrive verb together with it's subject and object from a C&C parsed file.

    File format description [1].

    [1] http://svn.ask.it.usyd.edu.au/trac/candc/wiki/MarkedUp
    """
    return Counter(_collect_verb_subject_object(f_name))


def _collect_verb_subject_object(f_name):
    logger.debug('Processing %s', f_name)

    with open(f_name, 'rt', encoding='utf8') as f:
        # Get rid of trailing whitespaces.
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

            dependencies = sorted(parse_dependencies(dependencies))
            tokens = dict(parse_tokens(c))

            for head_id, group in groupby(dependencies, lambda r: r[0]):
                group = list(group)

                try:
                    (_, obj, obj_id), (_, subj, subj_id) = sorted(g for g in group if g[1] in ('dobj', 'ncsubj'))
                except ValueError:
                    pass
                else:
                    if obj == 'dobj'and subj == 'ncsubj':

                        try:
                            yield tokens[head_id], tokens[subj_id], tokens[obj_id]
                        except KeyError:
                            logger.debug('Invalid group %s', group)
