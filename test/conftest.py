import gzip

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
def context_path(tmpdir):
    path = tmpdir.join('context.csv')

    context = pd.DataFrame(
        ['AA', 'BB', 'XX', 'YY', 'ZZ', 'aa', 'ab', 'ac', 'AA_NOUN', 'xx'],
        columns=('ngram', ),
    )

    context.to_csv(
        str(path),
        sep='\t',
        header=False,
        index=False,
    )

    return path


@pytest.fixture
def targets_path(tmpdir):
    path = tmpdir.join('targets.csv')

    path.write(
        'AA\n'  # 0
        'BB\n'  # 1
        'XX\n'  # 2
        'YY\n'  # 3
        'ZZ\n'  # 4
        'aa\n'  # 5
        'ab\n'  # 6
        'ac\n'  # 7
    )

    return path


@pytest.fixture
def cooccurrence_dir_path(tmpdir):
    path = tmpdir.join('cooccurrence')
    path.ensure_dir()

    with gzip.open(str(path.join('a_0.gz')), 'wt') as f:

        f.write(
            'AA\taa\t100\n'
            'BB\tac\t120\n'
            'AA\taa\t210\n'
            'AA\tac\t330\n'
            'xx\tac\t99\n'
        )

    with gzip.open(str(path.join('a_1.gz')), 'wt') as f:

        f.write(
            'XX\taa\t300\n'
            'YY\tac\t420\n'
            'ZZ\tab\t510\n'
            'AA\tac\t630\n'
            'AA_NOUN\tac\t630\n'
            'UNKNOWN\taa\t10000\n'
            'a\tUNKNOWN\t1000\n'
            '\uf8f0\tda\t333\n'
        )

    with gzip.open(str(path.join('a_2.gz')), 'wt') as f:

        f.write(
            'XX\taa\t1000\n'
        )

    return path
