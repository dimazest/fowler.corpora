import sys
import os

import pandas as pd
from fowler.corpora.models import read_space_from_file

import py
import pytest


@pytest.fixture(autouse=True, scope='session')
def pythonpath():
    os.environ['PYTHONPATH'] = ':'.join(sys.path)


@pytest.fixture
def datadir():
    return py.path.local(__file__).dirpath('data')


@pytest.fixture
def bnc_path(datadir):
    return 'bnc://{}'.format(datadir.join('BNC', 'Texts'))


@pytest.fixture
def bnc_ccg_path(datadir):
    return 'bnc-ccg://{}'.format(datadir.join('CCG_BNC_v1'))


@pytest.fixture
def ukwac_path(datadir):
    return 'ukwac://{}'.format(datadir.join('WaCky'))


@pytest.fixture
def brown_path(datadir):
    return 'brown://'


@pytest.fixture
def dispatcher():
    from fowler.corpora.main import dispatcher_factory

    return dispatcher_factory()


@pytest.fixture(
    params=[
        '-j1',
        # ''
    ],
)
def workers(request):
    """The number of workers, e.g. `-j 1`."""
    return request.param


@pytest.fixture
def limit():
    """The number of corpus files to process, e.g ``--limit 100`."""
    return ''


@pytest.fixture
def dictionary_path(tmpdir, dispatcher, corpus, limit, workers):
    path = str(tmpdir.join('dictinary.h5'))
    dispatcher.dispatch(
        'bnc dictionary '
        '--corpus {corpus} '
        '{tfl} '
        '--stem '
        '-o {output} '
        '{limit} '
        '{workers} '
        ''.format(
            corpus=corpus,
            output=path,
            tfl='' if corpus.startswith('bnc') else '--tag_first_letter',
            limit=limit,
            workers=workers,
        ).split()
    )

    return path


@pytest.fixture
def dictionary(dictionary_path):
    df = pd.read_hdf(dictionary_path, key='dictionary')
    return df


@pytest.fixture
def indexed_dictionary(dictionary):
    return dictionary.set_index(['ngram', 'tag'])


@pytest.fixture
def tokens_path(dictionary, tmpdir):
    path = str(tmpdir.join('tokens.csv'))

    dictionary[['ngram', 'tag']].to_csv(
        path,
        index=False,
        encoding='utf-8',
    )

    return path


@pytest.fixture
def context_path(tokens_path):
    return tokens_path


@pytest.fixture
def target_path(tokens_path):
    return tokens_path


@pytest.fixture
def space_path(tmpdir, dispatcher, corpus, context_path, target_path, limit, workers):
    path = str(tmpdir.join('space.h5'))
    dispatcher.dispatch(
        'bnc cooccurrence '
        '--corpus {corpus} '
        '-t {target} '
        '-c {context} '
        '-o {output} '
        '{workers} '
        '{limit} '
        '{tfl} '
        '--stem '
        ''.format(
            corpus=corpus,
            target=target_path,
            context=context_path,
            output=path,
            tfl='' if corpus.startswith('bnc') else '--tag_first_letter',
            limit=limit,
            workers=workers,
        ).split()
    )

    return path


@pytest.fixture
def space(space_path):
    return read_space_from_file(space_path)
