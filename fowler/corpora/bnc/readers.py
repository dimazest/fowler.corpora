import logging
import os.path
from os import getcwd

from nltk.corpus.reader.bnc import BNCCorpusReader
import pandas as pd
from py.path import local

from .util import ccg_bnc_iter, count_cooccurrence

logger = logging.getLogger(__name__)


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

        return ccg_bnc_iter(path, word_tags)
