import csv
import gzip
import logging
import os.path

from collections import namedtuple
from itertools import chain, takewhile, groupby, product
from os import getcwd

import pandas as pd

from more_itertools import peekable
from py.path import local

from nltk.corpus import brown, CategorizedTaggedCorpusReader
from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.parse.dependencygraph import DependencyGraph, DependencyGraphError
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem.snowball import SnowballStemmer



logger = logging.getLogger(__name__)


Token = namedtuple('Token', 'word, stem, tag')
Dependency = namedtuple('Dependency', 'head, relation, dependant')


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


class SingleFileDatasetMixIn:

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac'):

        if root is None:
            root = os.path.join(getcwd(), cls.default_file_name)

        return {
            'paths': [root],
            'tagset': tagset,
        }


class KS13(SingleFileDatasetMixIn):
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'
    default_file_name = 'emnlp2013_turk.txt'

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


class PhraseRel(SingleFileDatasetMixIn):
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'
    extra_fields = 'relevance_type',
    default_file_name = 'phraserel.csv'

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


class GS11(SingleFileDatasetMixIn):
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
    default_file_name = 'GS2011data.txt'

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


class GS12(SingleFileDatasetMixIn):
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'
    default_file_name = 'GS2012data.txt'

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


class SimLex999(SingleFileDatasetMixIn):
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'lexical'
    default_file_name = 'SimLex-999.txt'

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


class MEN(SingleFileDatasetMixIn):
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'lexical'
    default_file_name = 'MEN_dataset_lemma_form_full'

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


class MSRParaphraseCorpus():
    # TODO: Corpus readers should define tag mapping!

    vectorizer = 'compositional'
    extra_fields = 'split',

    def __init__(self, paths, tagset):
        self.paths = paths
        self.tagset = tagset

    @classmethod
    def init_kwargs(cls, root=None, tagset='ukwac', split=None):

        if root is None:
            root = os.path.join(getcwd(), 'MSRParaphraseCorpus')

        if split is None:
            paths = [
                (os.path.join(root, 'msr_paraphrase_train.txt'), 'train'),
                (os.path.join(root, 'msr_paraphrase_test.txt'), 'test'),
            ]
        elif split == 'train':
            paths = [
                (os.path.join(root, 'msr_paraphrase_train.txt'), 'train'),
            ]
        elif split == 'test':
            paths = [
                (os.path.join(root, 'msr_paraphrase_test.txt'), 'test'),
            ]

        return {
            'paths': paths,
            'tagset': tagset,
        }

    def read_file(self):
        dfs = []
        for path, split in self.paths:
            df = pd.read_csv(
                path,
                sep='\t',
                quoting=csv.QUOTE_NONE,
                encoding='utf-8-sig',
            )
            df['split'] = split
            dfs.append(df)

        df = pd.concat(dfs)

        logger.warn('Replacng `Treasury\\x12s` with `Treasurys`.')
        df['#1 String'] = df['#1 String'].str.replace('Treasury\x12s', 'Treasurys')

        return df

    def words_by_document(self, path):
        # Part of CorpusReader

        def words_iter():
            for g1, g2, _ in self.dependency_graphs_pairs():
                for g in g1, g2:
                    for node in g.nodes.values():
                        if not node['address']:
                            continue

                        yield (
                            node['word'],
                            node['lemma'],
                            node['tag'],
                        )

        yield words_iter()

    def dependency_graphs_pairs(self):
        # Part of Dataset
        from fowler.corpora.wsd.datasets import tag_mappings

        df = self.read_file()
        parser = StanfordDependencyParser()

        def parse(string):
            dg = next(parser.raw_parse(string))

            for node in dg.nodes.values():
                if not node['address']:
                    continue

                node['original_tag'] = node['tag']
                node['tag'] = tag_mappings[self.tagset][node['tag'][0]]

            return dg

        for _, row in df.iterrows():
            yield (
                parse(row['#1 String']),
                parse(row['#2 String']),
                row['Quality'],
                row['split'],
            )


class ANDailment(SingleFileDatasetMixIn):

    vectorizer = 'compositional'
    default_file_name = 'ANDailment.cvs'

    def read_file(self):
        # TODO: should be moved away from here.
        from fowler.corpora.wsd.datasets import tag_mappings

        df = pd.read_csv(
            self.paths[0],
        )

        def extract_head(row, side, argument_type):
            parser = StanfordDependencyParser()

            argument = row[side].split(
                row['rule_{}'.format(side)]
            )[0 if argument_type == 'subj' else 1]

            dg = next(parser.raw_parse(argument))
            return dg.root['lemma']

        df['lhs_subj'] = df.apply(extract_head, args=('lhs', 'subj'), axis=1)
        df['lhs_obj'] = df.apply(extract_head, args=('lhs', 'obj'), axis=1)

        df['rhs_subj'] = df.apply(extract_head, args=('rhs', 'subj'), axis=1)
        df['rhs_obj'] = df.apply(extract_head, args=('rhs', 'obj'), axis=1)

        for item, tag in (
            ('lhs_subj', 'N'),
            ('rhs_subj', 'N'),
            ('rule_lhs', 'V'),
            ('rule_rhs', 'V'),
            ('lhs_obj', 'N'),
            ('rhs_obj', 'N'),
        ):
            df['{}_tag'.format(item)] = tag_mappings[self.tagset][tag]

        return df

    def words_by_document(self, path):
        # Part of CorpusReader
        df = self.read_file()

        def words_iter(rows):
            for _, row in rows:
                for item in (
                    'lhs_subj', 'rhs_subj', 'lhs_obj', 'rhs_obj',
                ):
                    word = row[item]
                    stem = word.lower()
                    t = row['{}_tag'.format(item)]
                    yield word, stem, t

        yield words_iter(df.iterrows())

    def dependency_graphs_pairs(self):
        # Part of Dataset
        df = self.read_file()

        # DOTO: get rid of .lower()
        for _, row in df.iterrows():
            yield (
                self.sentence_to_graph(
                    row['lhs_subj'].lower(), row['lhs_subj_tag'],
                    row['rule_lhs'].lower(), row['rule_lhs_tag'],
                    row['lhs_obj'].lower(), row['lhs_obj_tag'],
                ),
                self.sentence_to_graph(
                    row['rhs_subj'].lower(), row['rhs_subj_tag'],
                    row['rule_rhs'].lower(), row['rule_rhs_tag'],
                    row['rhs_obj'].lower(), row['rhs_obj_tag'],
                ),
                row['entails']
            )

    def sentence_to_graph(self, s, s_t, v, v_t, o, o_t):
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
            ),
            cell_separator='\t',
        )
