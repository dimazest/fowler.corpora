import numpy as np
from fowler.corpora.models import read_space_from_file

import pytest


@pytest.fixture
def corpus(brown_path):
    return brown_path


@pytest.mark.parametrize(
    ('limit', 'counts'),
    (
        (
            '--limit 1',
            (
                (('sandman', 'N'), None),
                (('kelsey', 'N'), None),
            ),
        ),
        (
            '--limit 12',
            (
                (('sandman', 'N'), None),
                (('kelsey', 'N'), 2),
                (('statement', 'N'), 17),
            ),
        ),
        (
            '--limit 16',
            (
                (('sandman', 'N'), None),
                (('kelsey', 'N'), 2),
            ),
        ),
        (
            '--limit 17',
            (
                (('sandman', 'N'), None),
                (('kelsey', 'N'), 6),
            ),
        ),
        (
            '--limit 30',
            (
                (('gain', 'V'), 7),
                (('gain', 'N'), 14),
                (('for', 'I'), 687),
                (('cryptic', 'J'), None),
                (('sandman', 'N'), 5),
                (('kelsey', 'N'), 6),
            ),
        ),
        (
            '--limit 45',  # All ca* files.
            (
                (('sandman', 'N'), 5),
                (('kelsey', 'N'), 6),
            ),
        ),
        (
            '--limit 83',
            (
                (('cryptic', 'J'), None),
                (('sandman', 'N'), 5),
                (('kelsey', 'N'), 6),
                (('statement', 'N'), 44),
                (('zurich', 'N'), 2),
            ),
        ),
        (
            '--limit 84',
            (
                (('cryptic', 'J'), None),
                (('sandman', 'N'), 5),
                (('kelsey', 'N'), 6),
                (('statement', 'N'), 44),
            ),
        ),
        (
            '',
            (
                (('the', 'A'), 69968),
                (('for', 'I'), 8987),
                (('cryptic', 'J'), 3),
                (('sandman', 'N'), 5),
                (('kelsey', 'N'), 6),
                (('zurich', 'N'), 2),
            ),
        ),
    )
)
def test_dictionary(indexed_dictionary, counts):
    for token, count in counts:
        if count is not None:
            assert indexed_dictionary.loc[token, 'count'] == count
        else:
            assert token not in indexed_dictionary


@pytest.mark.parametrize(
    ('limit', 'values'),
    (
        (
            '',
            (),
        ),
        (
            '--limit 2',
            (
                (('statement', ('provid', 'V')), 0),
                (('statement', ('water', 'N')), 1),
                (('statement', ('for', 'I')), 1),
                (('statement', ('rural', 'J')), 1),
                (('statement', ('texa', 'N')), 1),
                (('statement', ('.', '.')), 1),

                (('statement', ('by', 'I')), 1),
                (('statement', ('other', 'A')), 1),
                (('statement', ('legisl', 'N')), 1),
                (('statement', ('that', 'C')), 1),
                (('statement', ('dalla', 'N')), 1),
                (('statement', ('is', 'B')), 0),
            ),
        ),
        (
            '--limit 83',
            (),
        ),
        (
            '--limit 84',
            (),
        ),
    )
)
def test_space(space, indexed_dictionary, values):
    for key, value in values:
        assert space.value(*key) == value

    row_labels = space.row_labels.sort('id')
    dict_counts = indexed_dictionary.loc[row_labels.index]
    dict_counts = dict_counts.values * 10
    space_totals = space.matrix.sum(axis=1)

    violations = space_totals > dict_counts

    assert (~violations).all()


@pytest.fixture
def conditional_space_path(space_path, tmpdir, dictionary_path, dispatcher):
    path = str(tmpdir.join('conditional_space.h5'))
    dispatcher.dispatch(
        'space pmi '
        '--dictionary {dictionary_path} '
        '-s {space_path} '
        '-o {output} '
        '--conditional-probability '
        '--remove-missing '
        ''.format(
            dictionary_path=dictionary_path,
            space_path=space_path,
            output=path,
        ).split()
    )

    return path


@pytest.fixture
def conditional_space(conditional_space_path):
    return read_space_from_file(conditional_space_path)


def test_conditional_space(conditional_space):
    violations = conditional_space.matrix > 1
    assert len(np.argwhere(violations)) == 0
