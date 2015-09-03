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
