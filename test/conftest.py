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
