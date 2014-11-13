import logging
import os.path
from os import getcwd
from collections import namedtuple
from itertools import chain, takewhile, groupby

import pandas as pd

from nltk.corpus.reader.bnc import BNCCorpusReader
from py.path import local

from .util import count_cooccurrence


logger = logging.getLogger(__name__)


CCGToken = namedtuple('CCGToken', 'word, stem, tag')
CCGDependency = namedtuple('CCGDependency', 'head, relation, dependant')


class Corpus:
    def __init__(self, paths, stem, tag_first_letter):
        self.paths = paths
        self.stem = stem
        self.tag_first_letter = tag_first_letter

    def cooccurrence(self, args):
        """Count word co-occurrence in a corpus file."""
        path, window_size, targets, context = args

        logger.debug('Processing %s', path)

        counts = count_cooccurrence(self.words_iter(path), window_size=window_size)

        def join_columns(frame, prefix):
            # Targets or contexts might be just words, not (word, POS) pairs.
            if isinstance(frame.index[0], tuple):
                return prefix, '{}_tag'.format(prefix)

            return (prefix, )

        counts = counts.merge(targets, left_on=join_columns(targets, 'target'), right_index=True, how='inner')

        counts = counts.merge(
            context,
            left_on=join_columns(context, 'context'),
            right_index=True,
            how='inner',
            suffixes=('_target', '_context'),
        )[['id_target', 'id_context', 'count']]

        # XXX make sure that there are no duplicates!

        return counts

    def words(self, path):
        """Count all the words from a corpus file."""
        logger.debug('Processing %s', path)

        result = pd.DataFrame(self.words_iter(path), columns=('ngram', 'tag'))
        result['count'] = 1

        return result.groupby(('ngram', 'tag'), as_index=False).sum()

    def dependencies(self, path):
        logger.debug('Processing %s', path)

        result = pd.DataFrame(
            (
                (h.word, h.stem, h.tag, r, d.word, d.stem, d.tag)
                for h, r, d in self.dependencies_iter(path)
            ),
            columns=(
                'head_word',
                'head_stem',
                'head_tag',
                'relation',
                'dependant_word',
                'dependant_stem',
                'dependant_tag',
            )
        )

        if self.tag_first_letter:
            raise NotImplemented('Tagging by first letter is not supported!')

        if self.stem:
            group_coulumns = ('head_stem', 'head_tag', 'relation', 'dependant_stem', 'dependant_tag')
        else:
            group_coulumns = ('head_word', 'head_tag', 'relation', 'dependant_word', 'dependant_tag')

        result['count'] = 1

        return result.groupby(group_coulumns).sum()


class BNC(Corpus):
    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)

        self.root = root

    @classmethod
    def init_kwargs(cls, stem, tag_first_letter, root=None, **kwargs):
        if root is None:
            root = os.path.join(getcwd(), 'corpora', 'BNC', 'Texts')

        if 'fileids' not in kwargs:
            kwargs['fileids'] = r'[A-K]/\w*/\w*\.xml'

        return dict(
            root=root,
            paths=BNCCorpusReader(root=root, **kwargs).fileids(),
            stem=stem,
            tag_first_letter=tag_first_letter,
        )

    def words_iter(self, path):
        for word, tag in BNCCorpusReader(fileids=path, root=self.root).tagged_words(stem=self.stem):
            if self.tag_first_letter:
                tag = tag[0]

            yield word, tag


class BNC_CCG(Corpus):

    @classmethod
    def init_kwargs(cls, stem, tag_first_letter, root=None, **kwargs):
        if root is None:
            root = os.path.join(getcwd(), 'corpora', 'CCG_BNC_v1')

        return dict(
            paths=[str(n) for n in local(root).visit() if n.check(file=True, exists=True)],
            stem=stem,
            tag_first_letter=tag_first_letter,
        )

    def words_iter(self, path):
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

                dependencies = self.parse_dependencies(dependencies)

                yield from postprocessor(dependencies, tokens)

    def collect_verb_subject_object(self, path):
        """Retrieve verb together with it's subject and object from a C&C parsed file.

        File format description [1] or Table 13 in [2].

        [1] http://svn.ask.it.usyd.edu.au/trac/candc/wiki/MarkedUp
        [2] http://anthology.aclweb.org/J/J07/J07-4004.pdf

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
        dependencies = sorted(
            d for d in dependencies if d.relation in ('dobj', 'ncsubj')
        )

        for head_id, group in groupby(dependencies, lambda d: d.head):
            group = list(group)

            try:
                (_, obj, obj_id), (_, subj, subj_id) = sorted(g for g in group if g.relation in ('dobj', 'ncsubj'))
            except ValueError:
                pass
            else:
                if obj == 'dobj'and subj == 'ncsubj':

                    try:
                        yield tuple(chain(tokens[head_id], tokens[subj_id], tokens[obj_id]))
                    except KeyError:
                        logger.debug('Invalid group %s', group)

    def dependencies_iter(self, path):

        def collect_dependencies(dependencies, tokens):
            for d in dependencies:
                yield CCGDependency(tokens[d.head], d.relation, tokens[d.dependant])


        return self.ccg_bnc_iter(path, collect_dependencies)

    def parse_dependencies(self, dependencies):
        """Parse and filter out verb subject/object dependencies from a C&C parse."""
        for dependency in dependencies:
            assert dependency[0] == '('
            assert dependency[-1] == ')'
            dependency = dependency[1:-1]

            split_dependency = dependency.split()
            split_dependency_len = len(split_dependency)

            if split_dependency_len == 3:
                # (dobj in_15 judgement_17)
                relation, head, dependant = split_dependency
            elif split_dependency_len == 4:
                empty = lambda r: r == '_' or '_' not in r
                if empty(split_dependency[-1]):
                    # (ncsubj being_19 judgement_17 _)
                    # (ncsubj laid_13 rule_12 obj)
                    relation, head, dependant = split_dependency[:-1]
                elif empty(split_dependency[1]):
                    # (xmod _ judgement_17 as_18)
                    # (ncmod poss CHOICE_4 IT_1)
                    relation, _ ,head, dependant = split_dependency
                else:
                    # (cmod who_11 people_3 share_12)
                    logger.debug('Ignoring dependency: %s', dependency)
                    continue

            else:
                logger.debug('Invalid dependency: %s', dependency)
                continue

            parse_argument = lambda a: int(a.split('_')[1])
            try:
                head_id = parse_argument(head)
                dependant_id = parse_argument(dependant)
            except (ValueError, IndexError):
                logger.debug('Could not extract dependency argument: %s', dependency)
                continue

            yield CCGDependency(head_id, relation, dependant_id)

    def parse_tokens(self, c):
        """Parse and retrieve token position, word, stem and tag from a C&C parse."""
        assert c[:4] == '<c> '
        c = c[4:]

        for position, token in enumerate(c.split()):
            word, stem, tag, *_ = token.split('|')

            yield position, CCGToken(word, stem, tag)
