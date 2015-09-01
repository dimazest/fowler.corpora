"""Access to the BNC corpus.

You can obtain the full version of the BNC corpus at
http://www.ota.ox.ac.uk/desc/2554

"""
import logging
import pickle

from urllib.parse import urlsplit, parse_qs

import pandas as pd

from pkg_resources import iter_entry_points
from progress.bar import Bar

from fowler.corpora.dispatcher import Dispatcher, NewSpaceCreationMixin, DictionaryMixin
from fowler.corpora.execnet import sum_folder
from fowler.corpora.space.util import write_space

from .readers import Corpus


logger = logging.getLogger(__name__)


def uri_to_corpus_reader(uri, workers_count=None, limit=None):
    corpus_uri = urlsplit(uri)
    query = parse_qs(corpus_uri.query)
    query_dict = {k: v[0] for k, v in query.items()}

    scheme_mapping = {
        ep.name: ep.load()
        for ep in iter_entry_points(group='fowler.corpus_readers', name=None)
    }

    try:
        CorpusReader = scheme_mapping[corpus_uri.scheme]
    except KeyError:
        raise NotImplementedError('The {0}:// scheme is not supported.'.format(corpus_uri.scheme))

    if corpus_uri.scheme == 'ukwac':
        query_dict['workers_count'] = workers_count

    corpus_reader_kwargs = CorpusReader.init_kwargs(
        root=corpus_uri.path or None,
        **query_dict
    )

    if limit:
        corpus_kwargs['paths'] = corpus_kwargs['paths'][:limit]

    return CorpusReader(**corpus_reader_kwargs)


class BNCDispatcher(Dispatcher, NewSpaceCreationMixin, DictionaryMixin):
    """BNC dispatcher."""
    global__corpus = '', 'bnc://', 'The path to the corpus.'
    global__stem = '', False, 'Use word stems instead of word strings.',
    global__tag_first_letter = '', False, 'Extract only the first letter of the POS tags.'
    global__window_size_before = '', 5, 'Window before.'
    global__window_size_after = '', 5, 'Window after.'

    @property
    def corpus(self):
        """Access to a corpus."""
        corpus_reader = uri_to_corpus_reader(
            uri=self.kwargs['corpus'],
            workers_count=len(self.execnet_hub.gateways),
            limit=self.limit
        )

        return Corpus(
            corpus_reader=corpus_reader,
            stem=self.stem,
            tag_first_letter=self.tag_first_letter,
            window_size_before=self.kwargs['window_size_before'],
            window_size_after=self.kwargs['window_size_after'],
        )

    @property
    def paths_progress_iter(self):
        paths = self.corpus.corpus_reader.paths

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
    execnet_hub,
    targets,
    context,
    paths_progress_iter,
    output=('o', 'space.h5', 'The output space file.'),
):
    """Build the co-occurrence matrix."""

    if targets.index.nlevels > 1:
        targets.sortlevel(inplace=True)
    if context.index.nlevels > 1:
        context.sortlevel(inplace=True)

    def init(channel):
        channel.send(
            (
                'data',
                pickle.dumps(
                    {
                        'kwargs': {
                            'targets': targets,
                            'context': context,
                        },
                        'instance': corpus,
                        'folder_name': 'cooccurrence',
                    },
                )
            )
        )

    result = execnet_hub.run(
        remote_func=sum_folder,
        iterable=paths_progress_iter,
        init_func=init,
    )

    result = [r for r in result if r is not None]

    result = pd.concat(result)
    result = result.groupby(['id_target', 'id_context']).sum()

    write_space(output, context, targets, result)


@command()
def dictionary(
    execnet_hub,
    corpus,
    dictionary_key,
    paths_progress_iter,
    omit_tags=('', False, 'Do not use POS tags.'),
    output=('o', 'dictionary.h5', 'The output file.'),
):
    """Count tokens."""
    def init(channel):
        channel.send(
            (
                'data',
                pickle.dumps(
                    {
                        'kwargs': {},
                        'instance': corpus,
                        'folder_name': 'words',
                    },
                )
            )
        )

    result = execnet_hub.run(
        remote_func=sum_folder,
        iterable=paths_progress_iter,
        init_func=init,
    )

    result = list(result)

    # This can be done in Corpus.words()
    if omit_tags:
        group_by = 'ngram',
    else:
        group_by = 'ngram', 'tag'

    result = pd.concat(result)
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
    execnet_hub,
    dictionary_key,
    corpus,
    paths_progress_iter,
    output=('o', 'transitive_verbs.h5', 'The output transitive verb file.'),
):
    """Count occurrence of transitive verbs together with their subjects and objects."""

    def init(channel):
        channel.send(
            (
                'data',
                pickle.dumps(
                    {
                        'instance': corpus,
                        'folder_name': 'verb_subject_object',
                    },
                )
            )
        )

    result = execnet_hub.run(
        remote_func=sum_folder,
        iterable=paths_progress_iter,
        init_func=init,
    )

    result = (
        pd.concat(r for r in result if r is not None)
        .groupby(
            ('verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag'),
            as_index=False,
        )
        .sum()
        .sort('count', ascending=False)
    )

    result.to_hdf(
        output,
        dictionary_key,
        mode='w',
        complevel=9,
        complib='zlib',
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
