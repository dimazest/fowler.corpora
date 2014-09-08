import py

import pytest

@pytest.fixture
def datadir():
    return py.path.local(__file__).dirpath('data')


@pytest.fixture
def bnc_path(datadir):
    return datadir.join('BNC', 'Texts')


@pytest.fixture
def dispatcher():
    from fowler.corpora.main import dispatcher

    return dispatcher


@pytest.fixture
def wordsim_base_path(datadir):
    return datadir.join('wordsim353')


@pytest.fixture
def wordsim_target_path(wordsim_base_path):
    return wordsim_base_path.join('targets_wordsim353.csv')

@pytest.fixture
def wordsim_context_path(wordsim_base_path):
    return wordsim_base_path.join('contexts_bnc_pos_1000.csv')
