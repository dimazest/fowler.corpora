import logging
from collections import namedtuple
from itertools import chain, takewhile, groupby

import pandas as pd

logger = logging.getLogger(__name__)


CCGToken = namedtuple('CCGToken', 'word, stem, tag')


class BNC_CCG:
    def __init__(self, paths, stem, tag_first_letter):
        self.paths = paths
        self.stem = stem
        self.tag_first_letter = tag_first_letter

    def words(self, path):
        def word_tags(dependencies, tokens):
            for token in tokens.values():

                tag = token.tag[0] if self.tag_first_letter else token.tag

                if self.stem:
                    yield token.stem, tag
                else:
                    yield token.word, tag

        return self.ccg_bnc_iter(path, word_tags)

    def ccg_bnc_iter(self, f_name, postprocessor):
        logger.debug('Processing %s', f_name)

        with open(f_name, 'rt', encoding='utf8') as f:
            # Get rid of trailing whitespace.
            lines = (l.strip() for l in f)

            while True:
                # Sentences are split by an empty line.
                sentence = list(takewhile(bool, lines))

                if not sentence:
                    # No line was taken, this means all the file has be read!
                    break

                # Take extra care of comments.
                sentence = [l for l in sentence if not l.startswith('#')]
                if not sentence:
                    # If we got nothing, but comments: skip.
                    continue

                *dependencies, c = sentence
                tokens = dict(self.parse_tokens(c))

                yield from postprocessor(dependencies, tokens)

    def collect_verb_subject_object(self, path):
        """Retrieve verb together with it's subject and object from a C&C parsed file.

        File format description [1].

        [1] http://svn.ask.it.usyd.edu.au/trac/candc/wiki/MarkedUp

        """
        columns = 'verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag'

        result = list(self.ccg_bnc_iter(path, self._collect_verb_subject_object))

        if result:
            result = pd.DataFrame(
                result,
                columns=columns,
            )
            result['count'] = 1

            return result.groupby(columns, as_index=False).sum()

    def _collect_verb_subject_object(self, dependencies, tokens):
        dependencies = sorted(self.parse_dependencies(dependencies))

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

    def parse_dependencies(self, dependencies):
        """Parse and filter out verb subject/object dependencies from a C&C parse."""
        for dependency in dependencies:
            assert dependency[0] == '('
            assert dependency[-1] == ')'
            dependency = dependency[1:-1]

            try:
                relation, head, dependant, *_ = dependency.split()
            except ValueError:
                logger.debug('Invalid dependency: %s', dependency)
                break

            if relation in set(['ncsubj', 'dobj']):
                yield (
                    int(head.split('_')[1]),
                    relation,
                    int(dependant.split('_')[1]),
                )

    def parse_tokens(self, c):
        """Parse and retrieve token position, word, stem and tag from a C&C parse."""
        assert c[:4] == '<c> '
        c = c[4:]

        for position, token in enumerate(c.split()):
            word, stem, tag, *_ = token.split('|')

            yield position, CCGToken(word, stem, tag)
