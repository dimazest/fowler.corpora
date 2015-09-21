import pytest

from fowler.corpora.models import read_space_from_file


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

    row_sums = normalized_space.matrix.sum(axis=1)

    for row_sum in row_sums[row_sums > 0].tolist()[0]:
        assert row_sum == 1.0

    assert normalized_space.matrix.sum() == 3.0


def test_truncate(space_path, space, tmpdir, dispatcher):
    path = str(tmpdir.join('truncated.h5'))

    dispatcher.dispatch(
        'space truncate '
        '-s {space_path} '
        '-o {output} '
        '--nvaa '
        '--tagset bnc '
        '--size 800'
        ''.format(
            space_path=space_path,
            output=path,
        ).split()
    )

    truncated = read_space_from_file(path)

    assert space.column_labels.loc['be', 'VERB']['id'] == 3
    assert space.column_labels.loc['have', 'VERB']['id'] == 10
    assert space.column_labels.loc['not', 'ADV']['id'] == 19
    assert space.column_labels.loc['this', 'ADJ']['id'] == 29
    assert space.column_labels.loc['do', 'VERB']['id'] == 30

    assert truncated.column_labels.loc['be', 'VERB']['id'] == 0
    assert truncated.column_labels.loc['have', 'VERB']['id'] == 1
    assert truncated.column_labels.loc['not', 'ADV']['id'] == 2
    assert truncated.column_labels.loc['this', 'ADJ']['id'] == 3
    assert truncated.column_labels.loc['do', 'VERB']['id'] == 4
