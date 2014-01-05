import py
import pandas as pd

import pytest


@pytest.fixture
def pytestbdd_feature_base_dir():
    here = py.path.local(__file__)
    return str(here.dirpath('..', 'features'))


@pytest.fixture
def datadir():
    return py.path.local(__file__).dirpath('data')


@pytest.fixture
def swda_100_path(datadir):
    return datadir.join('swda100.h5')


@pytest.yield_fixture
def swda_100_store(swda_100_path):
    with pd.get_store(str(swda_100_path), 'r') as store:
        yield store


@pytest.fixture
def wordsim_353_path(datadir):
    return datadir.join('wordsim353')


@pytest.fixture
def text_path(datadir):
    return datadir.join('gutenberg', 'pg16436.txt')


@pytest.fixture
def store_path(tmpdir):
    return tmpdir.join('store.h5')
