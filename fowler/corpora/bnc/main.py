"""Access to the BNC corpus.

You can obtain the full version of the BNC corpus at
http://www.ota.ox.ac.uk/desc/2554

"""
import logging
import os.path

from os import getcwd
from urllib.parse import urlsplit, parse_qs

import pandas as pd
from nltk.corpus.reader.bnc import BNCCorpusReader

from progress.bar import Bar
from py.path import local

from fowler.corpora.dispatcher import Dispatcher, NewSpaceCreationMixin, DictionaryMixin
from fowler.corpora.space.util import write_space

from .util import count_cooccurrence, collect_verb_subject_object, ccg_bnc_iter


logger = logging.getLogger(__name__)


class BNCDispatcher(Dispatcher, NewSpaceCreationMixin, DictionaryMixin):
    """BNC dispatcher."""
    global__corpus = '', 'bnc://', 'The path to the corpus.'
    global__stem = '', False, 'Use word stems instead of word strings.',
    global__tag_first_letter = '', False, 'Extract only the first letter of the POS tags.'

    @property
    def corpus(self):
        """Access to the corpus."""
        corpus = urlsplit(self.kwargs['corpus'])
        query = parse_qs(corpus.query)
        query_dict = {k: v[0] for k, v in query.items()}

        if corpus.scheme == 'bnc':
            if 'fileids' not in query_dict:
                query_dict['fileids'] = r'[A-K]/\w*/\w*\.xml'

            root = corpus.path or os.path.join(getcwd(), 'corpora', 'BNC', 'Texts')
            paths = BNCCorpusReader(root=root, **query_dict).fileids()

            Corpus = BNC
            corpus_kwargs = dict(
                root=root,
                stem=self.kwargs['stem'],
                tag_first_letter=self.kwargs['tag_first_letter'],
            )

        elif corpus.scheme == 'bnc+ccg':
            path = corpus.path or os.path.join(getcwd(), 'corpora', 'CCG_BNC_v1')
            paths = [str(n) for n in local(path).visit() if n.check(file=True, exists=True)]

            Corpus = BNC_CCG
            corpus_kwargs = dict(
                stem=self.kwargs['stem'],
                tag_first_letter=self.kwargs['tag_first_letter'],
            )
        else:
            raise NotImplementedError('The {0}:// scheme is not supported.'.format(corpus.scheme))

        if self.limit:
            paths = paths[:self.limit]

        return Corpus(paths=paths, **corpus_kwargs)

    @property
    def paths_progress_iter(self):
        paths = self.corpus.paths

        if not self.no_p11n:
            paths = Bar(
                'Reading the corpus',
                suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
            ).iter(paths)

        return paths

dispatcher = BNCDispatcher()
command = dispatcher.command


class BNC:
    def __init__(self, paths, root, stem, tag_first_letter):
        self.paths = paths
        self.root = root
        self.stem = stem
        self.tag_first_letter = tag_first_letter

    def words(self, path):
        for word, tag in BNCCorpusReader(fileids=path, root=self.root).tagged_words(stem=self.stem):
            if self.tag_first_letter:
                tag = tag[0]

            yield word, tag


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

        return ccg_bnc_iter(path, word_tags)


def corpus_cooccurrence(args):
    """Count word co-occurrence in a corpus file."""
    words, path, window_size, targets, context = args

    logger.debug('Processing %s', path)

    counts = count_cooccurrence(words(path), window_size=window_size)

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


@command()
def cooccurrence(
    corpus,
    pool,
    targets,
    context,
    paths_progress_iter,
    window_size=('', 5, 'Window size.'),
    output=('o', 'space.h5', 'The output space file.'),
):
    """Build the co-occurrence matrix."""
    matrices = (
        pool.imap_unordered(
            corpus_cooccurrence,
            ((corpus.words, path, window_size, targets, context) for path in paths_progress_iter),
        )
    )

    matrix = pd.concat((m for m in matrices)).groupby(['id_target', 'id_context']).sum()

    write_space(output, context, targets, matrix)


def corpus_words(args):
    """Count all the words from a corpus file."""
    words, path, = args

    logger.debug('Processing %s', path)

    result = pd.DataFrame(words(path), columns=('ngram', 'tag'))
    result['count'] = 1

    return result.groupby(('ngram', 'tag'), as_index=False).sum()


@command()
def dictionary(
    pool,
    corpus,
    dictionary_key,
    paths_progress_iter,
    omit_tags=('', False, 'Do not use POS tags.'),
    output=('o', 'dicitionary.h5', 'The output file.'),
):
    all_words = (
        pool.imap_unordered(
            corpus_words,
            ((corpus.words, path) for path in paths_progress_iter),
        )
    )

    if omit_tags:
        group_by = 'ngram',
    else:
        group_by = 'ngram', 'tag'

    (
        pd.concat(f for f in all_words if f is not None)
        .groupby(group_by)
        .sum()
        .sort('count', ascending=False)
        .reset_index()
        .to_hdf(
            output,
            dictionary_key,
            mode='w',
            complevel=9,
            complib='zlib',
        )
    )


@command()
def transitive_verbs(
    pool,
    dictionary_key,
    corpus,
    output=('o', 'transitive_verbs.h5', 'The output verb space file.'),
):
    """Count occurrence of transitive verbs together with their subjects and objects."""
    words, paths = corpus

    vsos = pool.imap_unordered(collect_verb_subject_object, paths)

    (
        pd.concat(f for f in vsos if f is not None)
        .groupby(
            ('verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag'),
            as_index=False,
        )
        .sum()
        .sort('count', ascending=False)
        .to_hdf(
            output,
            dictionary_key,
            mode='w',
            complevel=9,
            complib='zlib',
        )
    )
