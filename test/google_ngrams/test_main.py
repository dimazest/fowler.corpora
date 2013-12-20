import gzip
from fowler.corpora.main import dispatcher

import pytest


@pytest.fixture
def dictionary_path(tmpdir):
    path = tmpdir.join('dictionary.csv.gz')

    with gzip.open(str(path), 'wt') as f:
        f.write(
            'AA\t99\n'  # 0
            'BB\t90\n'  # 1
            'XX\t89\n'  # 2
            'YY\t88\n'  # 3
            'ZZ\t87\n'  # 4
            'aa\t80\n'  # 5
            'ab\t60\n'  # 6
            'ac\t50\n'  # 7
            'AA_NOUN\t3\n'  # 8
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


def test_cooccurrence(dictionary_path, cooccurrence_dir_path, capsys, tmpdir):
    output_path = tmpdir.join('cooccurrence.csv')

    dispatcher.dispatch(
        'google-ngrams cooccurrence '
        '--dictionary {dictionary_path} '
        '-i {cooccurrence_dir_path} '
        '-o {output_path}'
        ''.format(
            dictionary_path=dictionary_path,
            cooccurrence_dir_path=cooccurrence_dir_path,
            output_path=output_path,
        ).split()
    )

    out, _ = capsys.readouterr()

    assert len(out.split('\n')) == 4

    out = output_path.read().split('\n')
    assert len(out) == len(set(out))
    out = set(out)

    assert out == set(
        (
            '0\t5\t310',
            '0\t7\t960',
            '1\t7\t120',
            '2\t5\t1300',
            '3\t7\t420',
            '4\t6\t510',
            '',
        )
    )
