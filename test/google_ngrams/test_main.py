import gzip

import pandas as pd

from fowler.corpora.main import dispatcher

import pytest


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


def test_cooccurrence(context_path, cooccurrence_dir_path, capsys, tmpdir, targets_path):
    output_path = tmpdir.join('matrix.csv')

    dispatcher.dispatch(
        'google-ngrams cooccurrence '
        '--context {context_path} '
        '--targets {targets_path} '
        '-i {cooccurrence_dir_path} '
        '-o {output_path} '
        '-v '
        ''.format(
            context_path=context_path,
            cooccurrence_dir_path=cooccurrence_dir_path,
            output_path=output_path,
            targets_path=targets_path,
        ).split()
    )

    matrix = pd.read_hdf(str(output_path), 'matrix')
    assert len(matrix) == 6

    counts = matrix['count']

    assert counts.loc[0, 5] == 310
    assert counts.loc[0, 7] == 960
    assert counts.loc[1, 7] == 120
    assert counts.loc[2, 5] == 1300
    assert counts.loc[3, 7] == 420
    assert counts.loc[4, 6] == 510
