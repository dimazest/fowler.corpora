import py
import pandas as pd

import pytest


@pytest.fixture
def datadir():
    return py.path.local(__file__).dirpath().join('data')


@pytest.fixture
def swda_100_path(datadir):
    return datadir.join('swda100.h5')


@pytest.yield_fixture
def swda_100_store(swda_100_path):
    with pd.get_store(str(swda_100_path), 'r') as store:
        yield store
