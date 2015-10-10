import gzip
import logging
import os.path

from collections import namedtuple
from itertools import chain, takewhile, groupby, islice, count, product
from os import getcwd

import pandas as pd

from chrono import Timer
from more_itertools import peekable
from nltk.corpus import brown, CategorizedTaggedCorpusReader
from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.parse.dependencygraph import DependencyGraph, DependencyGraphError
from nltk.stem.snowball import SnowballStemmer
from py.path import local

from .util import co_occurrences


logger = logging.getLogger(__name__)


Token = namedtuple('Token', 'word, stem, tag')
Dependency = namedtuple('Dependency', 'head, relation, dependant')


class Corpus:

    # The peak memory usage per worker is about:
    #
    # * 3GB on the sentence similarity task based on ukWaC
    #   * 3k most frequent words as context
    #   * 185 target words
    #
    # * 4GB ITTF space based on ukWaC
    #   * 175154 context words!
    #   * 3k target words
    chunk_size = 1 * 10 ** 6

    def __init__(self, corpus_reader, stem, tag_first_letter, window_size_before, window_size_after):
        self.corpus_reader = corpus_reader

        self.stem = stem
        # TODO: `tag_first_letter` should become `shorten_tags`.
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

        target_contexts = peekable(chain.from_iterable(
                co_occurrences(
                    document_words,
                    window_size_before=self.window_size_before,
                    window_size_after=self.window_size_after,
                )
                for document_words in self.words_by_document(path)
            )
        )

        T = (lambda t: t) if isinstance(targets.index[0], tuple) else (lambda t: t[0])
        C = (lambda c: c) if isinstance(context.index[0], tuple) else (lambda c: c[0])

        first_frame, first_name = targets, 'target'
        second_frame, second_name = context, 'context'
        if len(context) < len(targets):
            first_frame, first_name, second_frame, second_name = (
                second_frame, second_name, first_frame, first_name
            )

        while target_contexts:
            some_target_contexts = islice(
                target_contexts,
                self.chunk_size,
            )

            with Timer() as timed:
                co_occurrence_pairs = list(some_target_contexts)

                if not co_occurrence_pairs:
                    continue

            logger.debug(
                '%s co-occurrence pairs: %.2f seconds',
                len(co_occurrence_pairs),
                timed.elapsed,
            )

            pairs = pd.DataFrame(
                co_occurrence_pairs,
                columns=columns,
            )

            def merge(pairs, what_frame, what_name, time, suffixes=None):
                kwargs = {'suffixes': suffixes} if suffixes else {}
                with Timer() as timed:
                    result = pairs.merge(
                        what_frame,
                        left_on=join_columns(what_frame, what_name),
                        right_index=True,
                        how='inner',
                        **kwargs
                    )
                logger.debug(
                    '%s merge (%s): %.2f seconds',
                    time,
                    what_name,
                    timed.elapsed,
                )

                return result

            pairs = merge(pairs, first_frame, first_name, 'First')
            pairs = merge(
                pairs,
                second_frame,
                second_name,
                'Second',
                suffixes=('_' + first_name, '_' + second_name),
            )

            with Timer() as timed:
                counts = pairs.groupby(['id_target', 'id_context']).size()
            logger.debug(
                'Summing up: %.2f seconds',
                timed.elapsed,
            )

            logger.debug(
                '%s unique co-occurrence pairs are collected. %s in total.',
                len(counts),
                counts.sum(),
            )

            yield counts

    def words_by_document(self, path):
        words_by_document = self.corpus_reader.words_by_document(path)

        for words in words_by_document:

            if self.stem:
                words = ((s, t) for _, s, t in words)
            else:
                words = ((w, t) for w, _, t in words)

            if self.tag_first_letter:
                words = ((n, t[0]) for n, t in words)

            yield words

    def words_iter(self, path):
        for document_words in self.words_by_document(path):
            yield from document_words

    def words(self, path):
        """Count all the words from a corpus file."""
        logger.debug('Processing %s', path)

        words = peekable(self.words_iter(path))
        iteration = count()
        while words:
            some_words = list(islice(words, self.chunk_size))

            logger.info('Computing frame #%s', next(iteration))

            # TODO: store all three values: ngram, stem and tag
            result = pd.DataFrame(
                some_words,
                columns=('ngram', 'tag'),
            )

            logger.debug('Starting groupby.')
            result = result.groupby(['ngram', 'tag']).size()
            logger.debug('Finished groupby.')

            yield result

    def dependencies(self, path):
        """Count dependency triples."""
        logger.debug('Processing %s', path)

        result = pd.DataFrame(
            (
                (h.word, h.stem, h.tag, r, d.word, d.stem, d.tag)
                for h, r, d in self.corpus_reader.dependencies_iter(path)
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

        result = list(self.corpus_reader.verb_subject_object_iter(path))
        if result:
            result = pd.DataFrame(
                result,
                columns=columns,
            )

            if self.tag_first_letter:
                for column in 'verb_tag', 'subj_tag', 'obj_tag':
                    result[column] = result[column].str.get(0)

            yield result.groupby(columns).size()


class BNC:
    def __init__(self, root, paths):
        self.root = root
        self.paths = paths

    @classmethod
    def init_kwargs(cls, root=None, fileids=r'[A-K]/\w*/\w*\.xml'):
        if root is None:
            root = os.path.join(getcwd(), 'BNC', 'Texts')

        return dict(
            root=root,
            paths=BNCCorpusReader(root=root, fileids=fileids).fileids(),
        )

    def words_by_document(self, path):
        def it():
            reader = BNCCorpusReader(fileids=path, root=self.root)
            words_tags = reader.tagged_words(stem=False)
            stems = (s for s, _ in reader.tagged_words(stem=True))

            for (word, tag), stem in zip(words_tags, stems):
                yield Token(word, stem, tag)

        # Consider the whole file as one document!
        yield it()


class Brown:
    def __init__(self, root, paths):
        self.root = root
        self.paths = paths

    @classmethod
    def init_kwargs(cls, root=None, fileids=None):
        return dict(
            root=brown.root if root is None else root,
            paths=brown.fileids() if fileids is None else fileids,
        )

    def words_by_document(self, path):
        stemmer = SnowballStemmer('english')

        def it():
            reader = CategorizedTaggedCorpusReader(
                fileids=[path],
                root=self.root,
                cat_file='cats.txt',
                tagset='brown',
                encoding='ascii',
            )

            for word, tag in reader.tagged_words():
                stem = stemmer.stem(word)
                yield Token(word, stem, tag)

        # Consider the whole file as one document!
        yield it()


class BNC_CCG:
    def __init__(self, paths):
        self.paths = paths

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
                # TODO: return a namedtuple?
                yield token.word, token.stem, token.tag

        # Consider the whole file as one document!
        for dependencies, tokens in self.ccg_bnc_iter(path):
            yield word_tags(dependencies, tokens)

    def ccg_bnc_iter(self, f_name):

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
                yield Dependency(tokens[d.head], d.relation, tokens[d.dependant])

        # Consider the whole file as one document!
        for dependencies, tokens in self.ccg_bnc_iter(path):
            yield from collect_dependencies(dependencies, tokens)

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

            yield Dependency(head_id, relation, dependant_id)

    def parse_tokens(self, c):
        """Parse and retrieve token position, word, stem and tag from a C&C parse."""
        assert c[:4] == '<c> '
        c = c[4:]

        for position, token in enumerate(c.split()):
            word, stem, tag, *_ = token.split('|')

            yield position, Token(word, stem, tag)


def ukwac_cell_extractor(cells):
    word, lemma, tag, feats, head, rel = cells
    return word, lemma, tag, tag, feats, head, rel


class UKWAC:

    def __init__(self, paths, file_passes, lowercase_stem, limit):
        self.paths = paths

        self.file_passes = int(file_passes)
        self.lowercase_stem = lowercase_stem
        self.limit = int(limit) if limit is not None else None

    @classmethod
    def init_kwargs(
        cls,
        root=None,
        workers_count=16,
        lowercase_stem=False,
        limit=None,
    ):
        if root is None:
            root = os.path.join(getcwd(), 'dep_parsed_ukwac')

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

        assert lowercase_stem in ('', 'y', False, True)
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
                if not l.startswith('<text') and l != '<s>'
            )

            c = 0
            while lines:
                if (c % (10 ** 4)) == 0:
                    logger.debug(
                        '%s text elements are read, every %s is processed. '
                        'It\'s about %.2f of the file.',
                        c,
                        self.file_passes,
                        c / 550000,  # An approximate number of texts in a file.
                    )

                if (self.limit is not None) and (c > self.limit):
                    logger.info('Limit of sentences is reached.')
                    break

                document = list(takewhile(lambda l: l != '</text>', lines))

                if (c % self.file_passes) == file_pass:
                    yield document

                c += 1

    def document_words(self, document):
        for dg in self.document_dependency_graphs(document):
            # Make sure that nodes are sorted by the position in the sentence.
            for _, node in sorted(dg.nodes.items()):

                if node['word'] is not None:
                    yield self.node_to_token(node)

    def verb_subject_object_iter(self, path):
        for document in self.documents(path):
            for dg in self.document_dependency_graphs(document):
                for node in dg.nodes.values():
                    if node['tag'][0] == 'V':
                        if 'SBJ' in node['deps'] and 'OBJ' in node['deps']:

                            for sbj_address, obj_address in product(
                                node['deps']['SBJ'],
                                node['deps']['OBJ'],
                            ):

                                sbj = dg.nodes[sbj_address]
                                obj = dg.nodes[obj_address]

                                yield (
                                    node['word'],
                                    node['lemma'],
                                    node['tag'],
                                    sbj['word'],
                                    sbj['lemma'],
                                    sbj['tag'],
                                    obj['word'],
                                    obj['lemma'],
                                    obj['tag'],
                                )

    def document_dependency_graphs(self, document):
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

            try:
                dg = DependencyGraph(
                    sentence,
                    cell_extractor=ukwac_cell_extractor,
                    cell_separator='\t',
                )
            except DependencyGraphError:
                logger.exception("Couldn't instantiate a dependency graph.")
            else:
                for node in dg.nodes.values():

                    if self.lowercase_stem and node['lemma']:
                        node['lemma'] = node['lemma'].lower()

                yield dg

    def node_to_token(self, node):
        return Token(node['word'], node['lemma'], node['tag'])

    def dependencies_iter(self, path):
        for document in self.documents(path):
            for dg in self.document_dependency_graphs(document):
                for node in dg.nodes.values():
                    if node['head'] is not None:
                        yield Dependency(
                            self.node_to_token(dg.nodes[node['head']]),
                            node['rel'],
                            self.node_to_token(node)
                        )


class KS13:
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), 'emnlp2013_turk.txt')

        return {
            'paths': [root],
            'tagset': tagset,
        }

    def read_file(self, group=False):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
            sep=' ',
            usecols=(
                'subject1', 'verb1', 'object1',
                'subject2', 'verb2', 'object2',
                'score',
            ),
        )

        for item, tag in (
            ('subject1', 'N'),
            ('verb1', 'V'),
            ('object1', 'N'),
            ('subject2', 'N'),
            ('verb2', 'V'),
            ('object2', 'N'),
        ):
            df['{}_tag'.format(item)] = tag_mappings[self.tagset][tag]

        if group:
            df = df.groupby(
                [
                    'subject1', 'subject1_tag', 'verb1', 'verb1_tag', 'object1', 'object1_tag',
                    'subject2', 'subject2_tag', 'verb2', 'verb2_tag', 'object2', 'object2_tag',
                ],
                as_index=False,
            ).mean()

        return df

    def words_by_document(self, path):
        # Part of CorpusReader
        df = self.read_file()

        def words_iter(rows):
            for _, row in rows:
                for item in (
                    'subject1', 'verb1', 'object1',
                    'subject2', 'verb2', 'object2',
                ):
                    word = stem = row[item]
                    t = row['{}_tag'.format(item)]
                    yield word, stem, t

        yield words_iter(df.iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file(group=True)

        for _, row in df.iterrows():
            yield (
                transitive_sentence_to_graph(
                    row['subject1'], row['subject1_tag'],
                    row['verb1'], row['verb1_tag'],
                    row['object1'], row['object1_tag'],
                ),
                transitive_sentence_to_graph(
                    row['subject2'], row['subject2_tag'],
                    row['verb2'], row['verb2_tag'],
                    row['object2'], row['object2_tag'],
                ),
                row['score']
            )


class PhraseRel:
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'
    extra_fields = 'relevance_type',

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), 'phraserel.csv')

        return {
            'paths': [root],
            'tagset': tagset,
        }

    def read_file(self):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
            sep=',',
            usecols=(
                'query_subject', 'query_verb', 'query_object',
                'document_subject', 'document_verb', 'document_object',
                'relevance_type', 'relevance_mean',
            ),
        )

        for item, tag in (
            ('query_subject', 'N'),
            ('query_verb', 'V'),
            ('query_object', 'N'),
            ('document_subject', 'N'),
            ('document_verb', 'V'),
            ('document_object', 'N'),
        ):
            df['{}_tag'.format(item)] = tag_mappings[self.tagset][tag]

        return df

    def words_by_document(self, path):
        # Part of CorpusReader
        df = self.read_file()

        def words_iter(rows):
            for _, row in rows:
                for item in (
                    'query_subject', 'query_verb', 'query_object',
                    'document_subject', 'document_verb', 'document_object',
                ):
                    word = stem = row[item]
                    t = row['{}_tag'.format(item)]
                    yield word, stem, t

        yield words_iter(df.iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file()

        for _, row in df.iterrows():
            yield (
                transitive_sentence_to_graph(
                    row['query_subject'], row['query_subject_tag'],
                    row['query_verb'], row['query_verb_tag'],
                    row['query_object'], row['query_object_tag'],
                ),
                transitive_sentence_to_graph(
                    row['document_subject'], row['document_subject_tag'],
                    row['document_verb'], row['document_verb_tag'],
                    row['document_object'], row['document_object_tag'],
                ),
                row['relevance_mean'],
                row['relevance_type'],
            )


def transitive_sentence_to_graph(s, s_t, v, v_t, o, o_t):
    template = (
        '{s}\t{s_t}\t2\tSBJ\n'
        '{v}\t{v_t}\t0\tROOT\n'
        '{o}\t{o_t}\t2\tOBJ\n'
    )

    return DependencyGraph(
        template.format(
            s=s, s_t=s_t,
            v=v, v_t=v_t,
            o=o, o_t=o_t,
        )
    )


class GS11:
    """Transitive sentence disambiguation dataset described in [1].

    The data is available at [2].

    [1] Grefenstette, Edward, and Mehrnoosh Sadrzadeh. "Experimental support
    for a categorical compositional distributional model of meaning."
    Proceedings of the Conference on Empirical Methods in Natural Language
    Processing. Association for Computational Linguistics, 2011.

    [2] http://www.cs.ox.ac.uk/activities/compdistmeaning/GS2011data.txt

    """
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), 'GS2011data.txt')

        return {
            'paths': [root],
            'tagset': tagset,
        }

    def read_file(self, group=False):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
            sep=' ',
            usecols=(
                'verb', 'subject', 'object', 'landmark', 'input',
            ),
        )

        for item, tag in (
            ('subject', 'N'),
            ('verb', 'V'),
            ('object', 'N'),
            ('landmark', 'V'),
        ):
            df['{}_tag'.format(item)] = tag_mappings[self.tagset][tag]

        if group:
            df = df.groupby(
                [
                    'subject', 'subject_tag', 'verb', 'verb_tag', 'object', 'object_tag',
                    'landmark', 'landmark_tag'
                ],
                as_index=False,
            ).mean()

        return df

    def words_by_document(self, path):
        # Part of CorpusReader
        df = self.read_file()

        def words_iter(rows):
            for _, row in rows:
                for item in (
                    'subject', 'verb', 'object', 'landmark'
                ):
                    word = stem = row[item]
                    t = row['{}_tag'.format(item)]
                    yield word, stem, t

        yield words_iter(df.iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file(group=True)

        for _, row in df.iterrows():
            yield (
                transitive_sentence_to_graph(
                    row['subject'], row['subject_tag'],
                    row['verb'], row['verb_tag'],
                    row['object'], row['object_tag'],
                ),
                transitive_sentence_to_graph(
                    row['subject'], row['subject_tag'],
                    row['landmark'], row['landmark_tag'],
                    row['object'], row['object_tag'],
                ),
                row['input']
            )


class GS12:
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), 'GS2011data.txt')

        return {
            'paths': [root],
            'tagset': tagset,
        }

    def read_file(self, group=False):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
            sep=' ',
            usecols=(
                'adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj', 'annotator_score'
            ),
        )

        for item, tag in (
            ('adj_subj', 'J'),
            ('subj', 'N'),
            ('verb', 'V'),
            ('adj_obj', 'J'),
            ('obj', 'N'),
            ('landmark', 'V'),
        ):
            df['{}_tag'.format(item)] = tag_mappings[self.tagset][tag]

        if group:
            df = df.groupby(
                [
                    'adj_subj', 'adj_subj_tag',
                    'subj', 'subj_tag',
                    'verb', 'verb_tag',
                    'adj_obj', 'adj_obj_tag',
                    'obj', 'obj_tag',
                    'landmark', 'landmark_tag'
                ],
                as_index=False,
            ).mean()

        return df

    def words_by_document(self, path):
        # Part of CorpusReader
        df = self.read_file()

        def words_iter(rows):
            for _, row in rows:
                for item in (
                    'adj_subj', 'subj', 'verb', 'adj_obj', 'obj', 'landmark'
                ):
                    word = stem = row[item]
                    t = row['{}_tag'.format(item)]
                    yield word, stem, t

        yield words_iter(df.iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file(group=True)

        for _, row in df.iterrows():
            yield (
                self.sentence_to_graph(
                    row['adj_subj'], row['adj_subj_tag'],
                    row['subj'], row['subj_tag'],
                    row['verb'], row['verb_tag'],
                    row['adj_obj'], row['adj_obj_tag'],
                    row['obj'], row['obj_tag'],
                ),
                self.sentence_to_graph(
                    row['adj_subj'], row['adj_subj_tag'],
                    row['subj'], row['subj_tag'],
                    row['landmark'], row['landmark_tag'],
                    row['adj_obj'], row['adj_obj_tag'],
                    row['obj'], row['obj_tag'],
                ),
                row['annotator_score']
            )

    def sentence_to_graph(self, sa, sa_t, s, s_t, v, v_t, oa, oa_t, o, o_t):
        template = (
            '{sa}\t{sa_t}\t2\tamod\n'
            '{s}\t{s_t}\t3\tSBJ\n'
            '{v}\t{v_t}\t0\tROOT\n'
            '{oa}\t{oa_t}\t2\tamod\n'
            '{o}\t{o_t}\t3\tOBJ\n'
        )

        return DependencyGraph(
            template.format(
                sa=sa, sa_t=sa_t,
                s=s, s_t=s_t,
                v=v, v_t=v_t,
                oa=oa, oa_t=oa_t,
                o=o, o_t=o_t,
            )
        )


class SimLex999:
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'lexical'

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), 'SimLex-999.txt')

        return {
            'paths': [root],
            'tagset': tagset,
        }

    def read_file(self):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
            sep='\t',
            usecols=('word1', 'word2', 'POS', 'SimLex999'),
        )

        df.loc[df['POS'] == 'N', 'POS'] = tag_mappings[self.tagset]['N']
        df.loc[df['POS'] == 'V', 'POS'] = tag_mappings[self.tagset]['V']
        df.loc[df['POS'] == 'A', 'POS'] = tag_mappings[self.tagset]['J']

        return df

    def words_by_document(self, path):
        # Part of CorpusReader

        def words_iter(rows):
            for _, row in rows:
                for item in ('word1', 'word2'):
                    word = stem = row[item]
                    t = row['POS']
                    yield word, stem, t

        yield words_iter(self.read_file().iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file()

        for _, row in df.iterrows():
            yield (
                self.sentence_to_graph(
                    row['word1'], row['POS'],
                ),
                self.sentence_to_graph(
                    row['word2'], row['POS'],
                ),
                row['SimLex999']
            )

    def sentence_to_graph(self, w, t):
        template = (
            '{w}\t{t}\t0\tROOT\n'
        )

        return DependencyGraph(template.format(w=w, t=t))


class MEN:
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'lexical'

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), 'MEN_dataset_lemma_form_full')

        return {
            'paths': [root],
            'tagset': tagset,
        }

    def read_file(self):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
            sep=' ',
            names=('token1', 'token2', 'score'),
        )

        def split(item):
            tag = 'tag{}'.format(item)
            word_token = df['token{}'.format(item)].str.split('-', expand=True)
            df['word{}'.format(item)] = word_token[0]
            df[tag] = word_token[1]

            df.loc[df[tag] == 'n', tag] = tag_mappings[self.tagset]['N']
            df.loc[df[tag] == 'v', tag] = tag_mappings[self.tagset]['V']
            df.loc[df[tag] == 'j', tag] = tag_mappings[self.tagset]['J']

        split('1')
        split('2')

        return df

    def words_by_document(self, path):
        # Part of CorpusReader

        def words_iter(rows):
            for _, row in rows:
                for item in ('1', '2'):
                    word = stem = row['word{}'.format(item)]
                    t = row['tag{}'.format(item)]
                    yield word, stem, t

        yield words_iter(self.read_file().iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file()

        for _, row in df.iterrows():
            yield (
                self.sentence_to_graph(
                    row['word1'], row['tag1'],
                ),
                self.sentence_to_graph(
                    row['word2'], row['tag2'],
                ),
                row['score']
            )

    def sentence_to_graph(self, w, t):
        template = (
            '{w}\t{t}\t0\tROOT\n'
        )

        return DependencyGraph(template.format(w=w, t=t))
