import py

import pytest


@pytest.fixture
def pytestbdd_feature_base_dir():
    here = py.path.local(__file__)
    return str(here.dirpath('..', 'features'))


@pytest.fixture
def datadir():
    return py.path.local(__file__).dirpath('data')


@pytest.fixture
def wordsim_353_path(datadir):
    return datadir.join('wordsim353')


@pytest.fixture
def context_path(datadir):
    return datadir.join('context.csv')


@pytest.fixture
def google_ngrams_path(datadir):
    return datadir.join('google_ngrams')


@pytest.fixture
def store_path(tmpdir):
    return tmpdir.join('store.h5')


@pytest.fixture
def cooccurrence_dir_path(google_ngrams_path):
    return google_ngrams_path.join('5_cooccurrence')


@pytest.fixture
def dispatcher():
    from fowler.corpora.main import dispatcher
    return dispatcher
