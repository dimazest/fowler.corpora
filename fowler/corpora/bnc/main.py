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

from .readers import BNC, BNC_CCG, UKWAC


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
            'dep-parsed-ukwac': UKWAC,
        }

        try:
            Corpus = scheme_mapping[corpus_uri.scheme]
        except KeyError:
            raise NotImplementedError('The {0}:// scheme is not supported.'.format(corpus_uri.scheme))

        corpus_kwargs = Corpus.init_kwargs(
            root=corpus_uri.path or None,
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
    window_size_before=('', 5, 'Window size before.'),
    window_size_after=('', 5, 'Window size after.'),
    output=('o', 'space.h5', 'The output space file.'),
):
    """Build the co-occurrence matrix."""
    window_size = window_size_before, window_size_after

    matrices = (
        pool.imap_unordered(
            corpus.cooccurrence,
            ((path, window_size, targets, context) for path in paths_progress_iter),
        )
    )

    matrix = pd.concat(
        (m for m in matrices if m is not None)
    ).groupby(['id_target', 'id_context']).sum()

    write_space(output, context, targets, matrix)


@command()
def dictionary(
    pool,
    corpus,
    dictionary_key,
    paths_progress_iter,
    omit_tags=('', False, 'Do not use POS tags.'),
    output=('o', 'dictionary.h5', 'The output file.'),
):
    """Count tokens."""
    word_chunks = pool.imap_unordered(corpus.words, paths_progress_iter)

    if omit_tags:
        group_by = 'ngram',
    else:
        group_by = 'ngram', 'tag'

    result = pd.concat(c for c in word_chunks if c is not None)
    assert result['count'].notnull().all()

    result = result.groupby(group_by).sum()
    assert result['count'].notnull().all()
    assert result.index.is_unique

    result.sort('count', ascending=False, inplace=True)
    result.reset_index(inplace=True)

    assert result.notnull().all().all()

    result.to_hdf(
        output,
        dictionary_key,
        mode='w',
        complevel=9,
        complib='zlib',
    )


@command()
def transitive_verbs(
    pool,
    dictionary_key,
    corpus,
    paths_progress_iter,
    output=('o', 'transitive_verbs.h5', 'The output verb space file.'),
):
    """Count occurrence of transitive verbs together with their subjects and objects."""
    vsos = pool.imap_unordered(corpus.collect_verb_subject_object, paths_progress_iter)

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


@command()
def dependencies(
    pool,
    dictionary_key,
    corpus,
    paths_progress_iter,
    output=('o', 'dependencies.h5', 'The output file.'),
):
    """Count dependencies (head, relation, dependant)."""
    dependencies = pd.concat(
        f for f in pool.imap_unordered(corpus.dependencies, paths_progress_iter) if f is not None
    )

    group_by = dependencies.index.names
    dependencies = (
        dependencies
        .reset_index()
        .groupby(group_by)
        .sum()
        .sort('count', ascending=False)
    )

    dependencies.to_hdf(
        output,
        dictionary_key,
        mode='w',
        complevel=9,
        complib='zlib',
    )
