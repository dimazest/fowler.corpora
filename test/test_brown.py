import numpy as np
from fowler.corpora.models import read_space_from_file

import pytest


@pytest.fixture
def corpus(brown_path):
    return brown_path


def test_dict(dictionary):
    dictionary.set_index(['ngram', 'tag'], inplace=True)

    assert dictionary.loc['the', 'A']['count'] == 69968
    assert dictionary.loc['for', 'I']['count'] == 8987


def test_space(space):
    assert space.value('she', ('window', 'N')) == 1
    assert space.value('she', ('door', 'N')) == 10

    assert space.value('he', ('window', 'N')) == 9
    assert space.value('he', ('door', 'N')) == 18


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


@pytest.mark.xfail(reason='Probabilities are evil!')
def test_conditional_space(conditional_space):
    violations = conditional_space.matrix > 1
    assert len(np.argwhere(violations)) == 0
