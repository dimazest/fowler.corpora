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
def wordsim_353_path(datadir):
    return datadir.join('wordsim353')


@pytest.fixture
def context_path(tmpdir):
    path = tmpdir.join('context.csv')

    context = pd.DataFrame(
        ['king', 'queen', 'drink', 'company', 'bishop', 'knows', 'send', 'loves', 'king_NOUN', 'tiger'],
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
        'king\n'     # 0
        'queen\n'    # 1
        'drink\n'    # 2
        'company\n'  # 3
        'bishop\n'   # 4
        'knows\n'    # 5
        'send\n'     # 6
        'loves\n'    # 7
    )

    return path


@pytest.fixture
def cooccurrence_dir_path(tmpdir):
    path = tmpdir.join('cooccurrence')
    path.ensure_dir()

    with gzip.open(str(path.join('a_0.gz')), 'wt') as f:

        f.write(
            'king\tknows\t100\n'
            'queen\tloves\t120\n'
            'king\tknows\t210\n'
            'king\tloves\t330\n'
            'tiger\tloves\t99\n'
        )

    with gzip.open(str(path.join('a_1.gz')), 'wt') as f:

        f.write(
            'drink\tknows\t300\n'
            'company\tloves\t420\n'
            'bishop\tsend\t510\n'
            'king\tloves\t630\n'
            'king_NOUN\tloves\t630\n'
            'UNKNOWN\tknows\t10000\n'
            'drink\tUNKNOWN\t1000\n'
            '\uf8f0\tda\t333\n'
        )

    with gzip.open(str(path.join('a_2.gz')), 'wt') as f:

        f.write(
            'drink\tknows\t1000\n'
        )

    return path
