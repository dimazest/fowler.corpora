import gzip
import logging
import os.path

from collections import namedtuple
from itertools import chain, takewhile, groupby, islice, count
from os import getcwd

import pandas as pd

from more_itertools import peekable
from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.parse.dependencygraph import DependencyGraph
from py.path import local

from .util import co_occurrences


logger = logging.getLogger(__name__)


CCGToken = namedtuple('CCGToken', 'word, stem, tag')
CCGDependency = namedtuple('CCGDependency', 'head, relation, dependant')


class Corpus:

    chunk_size = 1 * 10 ** 7

    def __init__(self, paths, stem, tag_first_letter, window_size_before, window_size_after):
        self.paths = paths
        self.stem = stem
        self.tag_first_letter = tag_first_letter
        self.window_size_before = window_size_before
        self.window_size_after = window_size_after

    def cooccurrence(self, path, targets, context):
        """Count word co-occurrence in a corpus file."""
        logger.debug('Processing %s', path)

        def join_columns(frame, prefix):
            # Targets or contexts might be just words, not (word, POS) pairs.
            if isinstance(frame.index[0], tuple):
                return prefix, '{}_tag'.format(prefix)

            return (prefix, )

        columns = 'target', 'target_tag', 'context', 'context_tag'

        target_contexts = peekable(
            co_occurrences(
                self.words_iter(path),
                window_size_before=self.window_size_before,
                window_size_after=self.window_size_after,
            )
        )

        while target_contexts:
            some_target_contexts = islice(
                target_contexts,
                self.chunk_size,
            )

            co_occurrence_pairs = list(
                chain.from_iterable(
                    (t + c for c in cs if c in context.index)
                    for t, cs in some_target_contexts
                    if t in targets.index
                )
            )

            if not co_occurrence_pairs:
                continue

            counts = pd.DataFrame(co_occurrence_pairs, columns=columns)
            counts['count'] = 1

            logger.debug('Merging contexts.')
            counts = counts.merge(
                context,
                left_on=join_columns(context, 'context'),
                right_index=True,
                how='inner',
            )

            logger.debug('Merging targets.')
            counts = counts.merge(
                targets,
                left_on=join_columns(targets, 'target'),
                right_index=True,
                how='inner',
                suffixes=('_context', '_target'),
            )[['id_target', 'id_context', 'count']]

            counts = counts.groupby(['id_target', 'id_context']).sum()
            logger.debug(
                '%s unique co-occurrence pairs are collected. %s in total.',
                len(counts),
                counts['count'].sum(),
            )
            yield counts

    def words_iter(self, path):
        NONE = None, None
        before = [NONE] * self.window_size_before
        after = [NONE] * self.window_size_after

        words_by_document = self.words_by_document(path)
        for document_words in words_by_document:
            yield from chain(before, document_words, after)

    def words(self, path):
        """Count all the words from a corpus file."""
        logger.debug('Processing %s', path)

        words = peekable(self.words_iter(path))
        iteration = count()
        while words:
            some_words = islice(words, self.chunk_size)

            logger.info('Computing frame #%s', next(iteration))
            result = pd.DataFrame(
                (x for x in some_words),
                columns=('ngram', 'tag'),
            )
            result['count'] = 1

            logger.debug('Starting groupby.')
            result = result.groupby(('ngram', 'tag')).sum()
            logger.debug('Finished groupby.')

            yield result

    def dependencies(self, path):
        """Count dependency triples."""
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

    def verb_subject_object(self, path):
        columns = 'verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag'

        result = list(self.verb_subject_object_iter(path))
        if result:
            result = pd.DataFrame(
                result,
                columns=columns,
            )
            result['count'] = 1

            return result.groupby(columns, as_index=False).sum()


class BNC(Corpus):
    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)

        self.root = root

    @classmethod
    def init_kwargs(cls, root=None, fileids=r'[A-K]/\w*/\w*\.xml'):
        if root is None:
            root = os.path.join(getcwd(), 'corpora', 'BNC', 'Texts')

        return dict(
            root=root,
            paths=BNCCorpusReader(root=root, fileids=fileids).fileids(),
        )

    def words_by_document(self, path):
        def it():
            for word, tag in BNCCorpusReader(fileids=path, root=self.root).tagged_words(stem=self.stem):
                if self.tag_first_letter:
                    tag = tag[0]

                yield word, tag

        # Consider the whole file as one document!
        yield it()


class BNC_CCG(Corpus):

    @classmethod
    def init_kwargs(cls, root=None):
        if root is None:
            root = os.path.join(getcwd(), 'corpora', 'CCG_BNC_v1')

        return dict(
            paths=[str(n) for n in local(root).visit() if n.check(file=True, exists=True)],
        )

    def words_by_document(self, path):
        def word_tags(dependencies, tokens):
            for token in tokens.values():

                tag = token.tag[0] if self.tag_first_letter else token.tag

                if self.stem:
                    yield token.stem, tag
                else:
                    yield token.word, tag

        # Consider the whole file as one document!
        for dependencies, tokens in self.ccg_bnc_iter(path):
            yield word_tags(dependencies, tokens)

    def ccg_bnc_iter(self, f_name):
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

                yield dependencies, tokens

    def verb_subject_object_iter(self, path):
        for dependencies, tokens in self.ccg_bnc_iter(path):
            yield from self._collect_verb_subject_object(dependencies, tokens)

    def _collect_verb_subject_object(self, dependencies, tokens):
        """Retrieve verb together with it's subject and object from a C&C parsed file.

        File format description [1] or Table 13 in [2].

        [1] http://svn.ask.it.usyd.edu.au/trac/candc/wiki/MarkedUp
        [2] http://anthology.aclweb.org/J/J07/J07-4004.pdf

        """

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

        # Consider the whole file as one document!
        for dependencies, tokens in self.ccg_bnc_iter(path):
            yield collect_dependencies(dependencies, tokens)

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
                    relation, _, head, dependant = split_dependency
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


def ukwac_cell_extractor(cells):
    word, lemma, tag, feats, head, rel = cells
    return word, lemma, tag, tag, feats, head, rel


class UKWAC(Corpus):

    def __init__(self, file_passes, lowercase_stem, limit, **kwargs):
        super().__init__(**kwargs)

        self.file_passes = int(file_passes)
        self.lowercase_stem = lowercase_stem
        self.limit = int(limit)

    @classmethod
    def init_kwargs(
        cls,
        root=None,
        workers_count=16,
        lowercase_stem=False,
        limit=None,
    ):
        if root is None:
            root = os.path.join(getcwd(), 'corpora', 'WaCky', 'dep_parsed_ukwac')

        paths = [
            str(n) for n in local(root).visit()
            if n.check(file=True, exists=True)
        ]

        file_passes = max(1, workers_count // len(paths))

        paths = list(
            chain.from_iterable(
                ((i, p) for p in paths)
                for i in range(file_passes)
            )
        )

        assert lowercase_stem in ('', 'y', False)
        lowercase_stem = bool(lowercase_stem)

        return dict(
            paths=paths,
            file_passes=file_passes,
            lowercase_stem=lowercase_stem,
            limit=limit,
        )

    def words_by_document(self, path):
        for document in self.documents(path):
            yield self.document_words(document)

    def documents(self, path):
        file_pass, path = path

        with gzip.open(path, 'rt', encoding='ISO-8859-1') as f:
            lines = (l.rstrip() for l in f)

            lines = peekable(
                l for l in lines
                if not l.startswith('<text')
                and l != '<s>'
            )

            c = 0
            while lines:
                if (c % (10 ** 4)) == 0:
                    logger.debug(
                        '%s text elements are read, every %s is processed.',
                        c,
                        self.file_passes,
                    )

                if (self.limit is not None) and (c > self.limit):
                    logger.info('Limit of sentences is reached.')
                    break

                document = list(takewhile(lambda l: l != '</text>', lines))

                if (c % self.file_passes) == file_pass:
                    yield document

                c += 1

    def document_words(self, document):
        for dg in self.dependency_graph_by_document(document):
            # Make sure that nodes are sorted by the position in the sentence.
            for _, node in sorted(dg.nodes.items()):
                ngram = node['lemma'] if self.stem else node['word']

                if ngram is not None:
                    if self.stem and self.lowercase_stem:
                        ngram = ngram.lower()

                    tag = node['tag']
                    if self.tag_first_letter:
                        tag = tag[0]

                    yield ngram, tag

    def dependency_graph_by_document(self, document):
        document = peekable(iter(document))

        while document:
            sentence = list(takewhile(lambda l: l != '</s>', document))

            if not sentence:
                # It might happen because of the snippets like this:
                #
                #    plates  plate   NNS     119     116     PMOD
                #    </text>
                #    </s>
                #    <text id="ukwac:http://www.learning-connections.co.uk/curric/cur_pri/artists/links.html">
                #    <s>
                #    Ideas   Ideas   NP      1       14      DEP
                #
                # where </text> is before </s>.
                continue

            yield DependencyGraph(
                sentence,
                cell_extractor=ukwac_cell_extractor,
                cell_separator='\t',
            )
