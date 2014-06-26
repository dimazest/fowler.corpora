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

            path = corpus.path or os.path.join(getcwd(), 'corpora', 'BNC', 'Texts')
            paths = BNCCorpusReader(root=path, **query_dict).fileids()

            words = bnc_words
            words_kwargs = {'root': path}
        elif corpus.scheme == 'bnc+ccg':
            path = corpus.path or os.path.join(getcwd(), 'corpora', 'CCG_BNC_v1')
            paths = [str(n) for n in local(path).visit() if n.check(file=True, exists=True)]

            words = bnc_ccg_words
            words_kwargs = {}
        else:
            raise NotImplementedError('The {0}:// scheme is not supported.')

        if self.limit:
            paths = paths[:self.limit]

        if not self.no_p11n:
            paths = Bar(
                'Reading the corpus',
                suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
            ).iter(paths)

        return words, paths, words_kwargs


dispatcher = BNCDispatcher()
command = dispatcher.command


def bnc_words(path, stem, tag_first_letter, **kwargs):
    for word, tag in BNCCorpusReader(fileids=path, **kwargs).tagged_words(stem=stem):
        if tag_first_letter:
            tag = tag[0]

        yield word, tag


def bnc_ccg_words(path, stem, tag_first_letter, **kwargs):
    def word_tags(dependencies, tokens):
        for token in tokens.values():
            if stem:
                yield token.stem, token.tag
            else:
                yield token.word, token.tag

    return ccg_bnc_iter(path, word_tags, tag_first_letter=tag_first_letter)


def corpus_cooccurrence(args):
    """Count word co-occurrence in a corpus file."""
    words, path, kwargs, tag_first_letter, window_size, stem, targets, context = args

    logger.debug('Processing %s', path)

    counts = count_cooccurrence(
        words(path, stem, tag_first_letter, **kwargs),
        window_size=window_size,
    )

    # Targets might be just words, not (word, POS) pairs.
    if isinstance(targets.index[0], tuple):
        target_join_columns = 'target', 'target_tag'
    else:
        target_join_columns = 'target',

    counts = (
        counts
        .merge(targets, left_on=target_join_columns, right_index=True, how='inner')
        .merge(
            context,
            left_on=('context', 'context_tag'),
            right_index=True,
            how='inner',
            suffixes=('_target', '_context'),
        )[['id_target', 'id_context', 'count']]
    )

    return counts


@command()
def cooccurrence(
    corpus,
    pool,
    targets,
    context,
    stem,
    tag_first_letter,
    window_size=('', 5, 'Window size.'),
    chunk_size=('', 7, 'Length of the chunk at the reduce stage.'),
    output=('o', 'matrix.h5', 'The output matrix file.'),
):
    """Build the co-occurrence matrix."""
    words, paths, kwargs = corpus

    matrices = (
        pool.imap_unordered(
            corpus_cooccurrence,
            ((words, path, kwargs, tag_first_letter, window_size, stem, targets, context) for path in paths),
        )
    )

    matrix = pd.concat(
        (m for m in matrices),
    ).groupby(['id_target', 'id_context']).sum()

    write_space(output, context, targets, matrix)


def corpus_words(args):
    """Count all the words from a corpus file."""
    words, path, kwargs, tag_first_letter, stem = args

    logger.debug('Processing %s', path)

    result = pd.DataFrame(
        words(path, stem, tag_first_letter, **kwargs),
        columns=('ngram', 'tag'),
    )
    result['count'] = 1

    return result.groupby(('ngram', 'tag'), as_index=False).sum()


@command()
def dictionary(
    pool,
    corpus,
    dictionary_key,
    tag_first_letter,
    stem=('', False, 'Use word stems instead of word strings.'),
    omit_tags=('', False, 'Do not use POS tags.'),
    output=('o', 'dicitionary.h5', 'The output file.'),
):
    words, paths, kwargs = corpus

    all_words = (
        pool.imap_unordered(
            corpus_words,
            ((words, path, kwargs, tag_first_letter, stem) for path in paths),
        )
    )

    (
        pd.concat(f for f in all_words if f is not None)
        .groupby(('ngram', 'tag'))
        .sum()
        .sort('count', ascending=False)
        .reset_index(drop=True)
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
    ccg_bnc,
    tag_first_letter,
    output=('o', 'transitive_verbs.h5', 'The output verb space file.'),
):
    """Count occurrence of transitive verbs together with their subjects and objects."""
    columns = 'verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag', 'count'
    vsos = pool.imap_unordered(
        collect_verb_subject_object,
        ((f, tag_first_letter) for f in ccg_bnc),
    )

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
