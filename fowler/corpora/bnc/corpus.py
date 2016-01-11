import logging

from itertools import chain, islice, count

import pandas as pd

from chrono import Timer
from more_itertools import peekable

from .util import co_occurrences

logger = logging.getLogger(__name__)


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
