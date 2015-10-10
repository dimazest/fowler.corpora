import numpy as np
import pytest

from fowler.corpora.models import read_space_from_file


@pytest.fixture
def corpus(bnc_path):
    return bnc_path


@pytest.fixture
def limit():
    return ''


def test_line_normalize(space_path, tmpdir, dispatcher):
    path = str(tmpdir.join('line-normalized.h5'))

    dispatcher.dispatch(
        'space line-normalize '
        '-s {space_path} '
        '-o {output} '
        ''.format(
            space_path=space_path,
            output=path,
        ).split()
    )

    normalized_space = read_space_from_file(path)

    violations = normalized_space.matrix > 1
    assert len(np.argwhere(violations)) == 0


def test_truncate(space_path, space, tmpdir, dispatcher):
    path = str(tmpdir.join('truncated.h5'))

    dispatcher.dispatch(
        'space truncate '
        '-s {space_path} '
        '-o {output} '
        '--nvaa '
        '--tagset bnc '
        '--size 40'
        ''.format(
            space_path=space_path,
            output=path,
        ).split()
    )

    truncated = read_space_from_file(path)

    assert space.column_labels.loc['be', 'VERB']['id'] == 3
    assert space.column_labels.loc['not', 'ADV']['id'] == 11
    assert space.column_labels.loc['do', 'VERB']['id'] == 16
    assert space.column_labels.loc['right', 'ADV']['id'] == 19
    assert space.column_labels.loc['first', 'ADJ']['id'] == 28
    assert space.column_labels.loc['have', 'VERB']['id'] == 61

    assert truncated.column_labels.loc['be', 'VERB']['id'] == 0
    assert truncated.column_labels.loc['not', 'ADV']['id'] == 3
    assert truncated.column_labels.loc['do', 'VERB']['id'] == 6
    assert truncated.column_labels.loc['right', 'ADV']['id'] == 8
    assert truncated.column_labels.loc['first', 'ADJ']['id'] == 14
    assert truncated.column_labels.loc['have', 'VERB']['id'] == 38
