import sys
import os

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
    return datadir.join('BNC', 'Texts')


@pytest.fixture
def bnc_ccg_path(datadir):
    return datadir.join('CCG_BNC_v1')


@pytest.fixture
def ukwac_path(datadir):
    return datadir.join('WaCky')


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


@pytest.fixture
def space_path(tmpdir, dispatcher, bnc_path, wordsim_target_path, wordsim_context_path):
    path = str(tmpdir.join("space.h5"))
    dispatcher.dispatch(
        'bnc cooccurrence '
        '--corpus bnc://{corpus} '
        '-t {target} '
        '-c {context} '
        '-o {output} '
        '--no_p11n'
        ''.format(
            corpus=bnc_path,
            target=wordsim_target_path,
            context=wordsim_context_path,
            output=path,
        ).split()
    )

    return path


@pytest.fixture
def space(space_path):
    return read_space_from_file(space_path)
