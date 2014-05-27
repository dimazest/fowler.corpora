"""Access to the BNC corpus.

You can obtain the full version of the BNC corpus at
http://www.ota.ox.ac.uk/desc/2554

"""
import logging

from collections import Counter
from itertools import chain

from more_itertools import chunked

import pandas as pd
from nltk.corpus.reader.bnc import BNCCorpusReader

from progress.bar import Bar
from py.path import local

from fowler.corpora.dispatcher import Dispatcher, Resource, NewSpaceCreationMixin, DictionaryMixin
from fowler.corpora.space.util import write_space

from .util import count_cooccurrence, collect_verb_subject_object, ccg_words


logger = logging.getLogger(__name__)


class BNCDispatcher(Dispatcher, NewSpaceCreationMixin, DictionaryMixin):
    """BNC dispatcher."""

    global__bnc = '', 'corpora/BNC/Texts', 'Path to the BNC corpus.'
    global__ccg_bnc = '', 'corpora/CCG_BNC_v1', 'Path to the CCG parsed BNC.'
    global__fileids = '', r'[A-K]/\w*/\w*\.xml', 'Files to be read in the corpus.'
    global__tag_first_letter = '', False, 'Extract only the first letter of the POS tags.'

    @Resource
    def bnc(self):
        """BNC corpus reader."""
        root = self.kwargs['bnc']
        return BNCCorpusReader(root=root, fileids=self.fileids)

    @property
    def ccg_bnc(self):
        files = [str(n) for n in local(self.kwargs['ccg_bnc']).visit() if n.check(file=True, exists=True)]

        if self.limit:
            files = files[:self.limit]

        if not self.no_p11n:
            files = Bar(
                'Reading CCG parsed BNC',
                suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
            ).iter(files)

        return files


dispatcher = BNCDispatcher()
command = dispatcher.command


def bnc_cooccurrence(args):
    """Count word co-occurrence in a BNC file."""
    root, fileids, window_size, stem, targets, context = args

    logger.debug('Processing %s', fileids)

    counts = count_cooccurrence(
        BNCCorpusReader(root=root, fileids=fileids).tagged_words(stem=stem),
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


def do_sum_counters(args):
    logger.debug('Summing up %d counters.', len(args))

    first_counter, *rest = args
    return sum(rest, first_counter)


def sum_counters(counters, pool, chunk_size=7):
    while True:
        counters = chunked(counters, chunk_size)

        first = next(counters)
        if len(first) == 1:
            logger.debug('Got results for a chunk.')
            return first[0]

        counters = pool.imap_unordered(do_sum_counters, chain([first], counters))


@command()
def cooccurrence(
    bnc,
    pool,
    targets,
    context,
    window_size=('', 5, 'Window size.'),
    chunk_size=('', 7, 'Length of the chunk at the reduce stage.'),
    stem=('', False, 'Use word stems instead of word strings.'),
    output=('o', 'matrix.h5', 'The output matrix file.'),
):
    """Build the co-occurrence matrix."""

    all_fileids = bnc.fileids()
    all_fileids = Bar(
        'Reading BNC',
        suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
    ).iter(all_fileids)

    matrices = (
        pool.imap_unordered(
            bnc_cooccurrence,
            ((bnc.root, fileids, window_size, stem, targets, context) for fileids in all_fileids),
        )
    )

    matrix = pd.concat(
        (m for m in matrices),
    ).groupby(['id_target', 'id_context']).sum()

    write_space(output, context, targets, matrix)


def bnc_words(args):
    root, fileids, c5, stem, omit_tags = args
    logger.debug('Processing %s', fileids)
    bnc = BNCCorpusReader(root=root, fileids=fileids)

    try:
        if not omit_tags:
            return Counter(bnc.tagged_words(stem=stem, c5=c5))
        else:
            return Counter(bnc.words(stem=stem))
    except:
        logger.error('Could not process %s', fileids)
        raise


@command()
def dictionary(
    bnc,
    pool,
    dictionary_key,
    output=('o', 'dicitionary.h5', 'The output file.'),
    c5=('', False, 'Use more detailed c5 tags.'),
    stem=('', False, 'Use word stems instead of word strings.'),
    omit_tags=('', False, 'Do not use POS tags.'),
):
    """Extract word frequencies from the corpus."""
    words = sum_counters(
        pool.imap_unordered(
            bnc_words,
            ((bnc.root, fid, c5, stem, omit_tags) for fid in bnc.fileids()),
        ),
        pool=pool,
    )

    logger.debug('The final counter contains %d items.', len(words))

    if not omit_tags:
        words = ([w, t, c] for (w, t), c in words.items())
        columns = 'ngram', 'tag', 'count'
    else:
        words = ([w, c] for w, c in words.items())
        columns = 'ngram', 'count'

    (
        pd.DataFrame(words, columns=columns)
        .sort('count', ascending=False)
        .to_hdf(
            output,
            dictionary_key,
            mode='w',
            complevel=9,
            complib='zlib',
        )
    )


@command()
def ccg_dictionary(
    pool,
    ccg_bnc,
    dictionary_key,
    tag_first_letter,
    stem=('', False, 'Use word stems instead of word strings.'),
    omit_tags=('', False, 'Do not use POS tags.'),
    output=('o', 'dicitionary.h5', 'The output file.'),
):
    all_words = pool.imap_unordered(
        ccg_words,
        ((f, tag_first_letter) for f in ccg_bnc),
    )

    frame = pd.concat(f for f in all_words if f is not None)

    if stem:
        groupby_columns = 'ngram', 'tag'
    else:
        groupby_columns = 'ngram', 'stem', 'tag'

    (
        frame
        .groupby(groupby_columns)
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
