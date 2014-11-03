"""Access to the BNC corpus.

You can obtain the full version of the BNC corpus at
http://www.ota.ox.ac.uk/desc/2554

"""
import logging
from urllib.parse import urlsplit, parse_qs

import pandas as pd

from progress.bar import Bar

from fowler.corpora.dispatcher import Dispatcher, NewSpaceCreationMixin, DictionaryMixin
from fowler.corpora.space.util import write_space

from .readers import BNC, BNC_CCG


logger = logging.getLogger(__name__)


class BNCDispatcher(Dispatcher, NewSpaceCreationMixin, DictionaryMixin):
    """BNC dispatcher."""
    global__corpus = '', 'bnc://', 'The path to the corpus.'
    global__stem = '', False, 'Use word stems instead of word strings.',
    global__tag_first_letter = '', False, 'Extract only the first letter of the POS tags.'

    @property
    def corpus(self):
        """Access to the corpus."""
        corpus_uri = urlsplit(self.kwargs['corpus'])
        query = parse_qs(corpus_uri.query)
        query_dict = {k: v[0] for k, v in query.items()}

        scheme_mapping = {
            'bnc': BNC,
            'bnc+ccg': BNC_CCG,
        }

        try:
            Corpus = scheme_mapping[corpus_uri.scheme]
        except KeyError:
            raise NotImplementedError('The {0}:// scheme is not supported.'.format(corpus_uri.scheme))

        corpus_kwargs = Corpus.init_kwargs(
            root=corpus_uri.path,
            stem=self.stem,
            tag_first_letter=self.tag_first_letter,
            **query_dict
        )

        if self.limit:
            corpus_kwargs['paths'] = corpus_kwargs['paths'][:self.limit]

        return Corpus(**corpus_kwargs)

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
            corpus.cooccurrence,
            ((path, window_size, targets, context) for path in paths_progress_iter),
        )
    )

    matrix = pd.concat((m for m in matrices)).groupby(['id_target', 'id_context']).sum()

    write_space(output, context, targets, matrix)


@command()
def dictionary(
    pool,
    corpus,
    dictionary_key,
    paths_progress_iter,
    omit_tags=('', False, 'Do not use POS tags.'),
    output=('o', 'dicitionary.h5', 'The output file.'),
):
    all_words = pool.imap_unordered(corpus.words, paths_progress_iter)

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
