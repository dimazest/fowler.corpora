import pandas as pd

from fowler.corpora.bnc.util import count_cooccurrence
from fowler.corpora.dispatcher import DictionaryMixin
from fowler.corpora.models import read_space_from_file

import pytest


@pytest.mark.parametrize(
    ('sequence', 'window_size', 'expected_result'),
    (
        (
            'abc',
            1,
            [
                ('a', 'b', 1),
                ('b', 'a', 1),
                ('b', 'c', 1),
                ('c', 'b', 1),
            ],
        ),
        (
            'abcdefg',
            2,
            [
                ('a', 'b', 1),
                ('a', 'c', 1),
                ('b', 'a', 1),
                ('b', 'c', 1),
                ('b', 'd', 1),
                ('c', 'a', 1),
                ('c', 'b', 1),
                ('c', 'd', 1),
                ('c', 'e', 1),
                ('d', 'b', 1),
                ('d', 'c', 1),
                ('d', 'e', 1),
                ('d', 'f', 1),
                ('e', 'c', 1),
                ('e', 'd', 1),
                ('e', 'f', 1),
                ('e', 'g', 1),
                ('f', 'd', 1),
                ('f', 'e', 1),
                ('f', 'g', 1),
                ('g', 'e', 1),
                ('g', 'f', 1),
            ],
        ),
        (
            'ab',
            100,
            [
                ('a', 'b', 1),
                ('b', 'a', 1),
            ],
        ),
        (
            'abbc',
            2,
            [
                ('a', 'b', 2),
                ('b', 'a', 2),
                ('b', 'b', 2),
                ('b', 'c', 2),
                ('c', 'b', 2),
            ],
        ),
    ),
)
def test_count_cooccurrence(sequence, window_size, expected_result):
    sequence = ((e, 'N') for e in sequence)

    result = count_cooccurrence(iter(sequence), window_size=window_size)

    expected_result = pd.DataFrame(expected_result, columns=('target', 'context', 'count'))
    expected_result['target_tag'] = 'N'
    expected_result['context_tag'] = 'N'
    expected_result = expected_result.reindex_axis(['target', 'target_tag', 'context', 'context_tag', 'count'], axis=1)

    assert (result == expected_result).all().all()


def test_bnc_dictionary(bnc_path, dispatcher, tmpdir):
    dictionary_path = str(tmpdir.join("dictionary.h5"))
    dispatcher.dispatch(
        'bnc dictionary '
        '--corpus bnc://{corpus} '
        '-o {output} '
        '--no_p11n'
        ''.format(
            corpus=bnc_path,
            output=dictionary_path,
        ).split()
    )

    dictionary = DictionaryMixin.get_dictionary(dictionary_path, 'dictionary')

    dictionary.set_index(['ngram', 'tag'], inplace=True)

    assert dictionary.loc[('.', 'PUN')]['count'] == 11
    assert dictionary.loc[('she', 'PRON')]['count'] == 9
    assert dictionary.loc[(',', 'PUN')]['count'] == 8
    assert dictionary.loc[('to', 'PREP')]['count'] == 7
    assert dictionary.loc[('and', 'CONJ')]['count'] == 5

    assert dictionary['count'].sum() == 151
    assert len(dictionary) == 88


def test_bnc_ccg_dictionary(bnc_ccg_path, dispatcher, tmpdir):
    dictionary_path = str(tmpdir.join("dictionary.h5"))
    dispatcher.dispatch(
        'bnc dictionary '
        '--corpus bnc+ccg://{corpus} '
        '-o {output} '
        '--no_p11n'
        ''.format(
            corpus=bnc_ccg_path,
            output=dictionary_path,
        ).split()
    )

    dictionary = DictionaryMixin.get_dictionary(dictionary_path, 'dictionary')

    dictionary.set_index(['ngram', 'tag'], inplace=True)

    assert dictionary.loc[('.', '.')]['count'] == 117
    assert dictionary.loc[('she', 'PRP')]['count'] == 1
    assert dictionary.loc[(',', ',')]['count'] == 55
    assert dictionary.loc[('to', 'TO')]['count'] == 39
    assert dictionary.loc[('and', 'CC')]['count'] == 22

    assert dictionary['count'].sum() == 1781
    assert len(dictionary) == 675


def test_bnc_cooccurrence(bnc_path, dispatcher, tmpdir, wordsim_target_path, wordsim_context_path):
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

    space = read_space_from_file(path)

    assert space.matrix.sum() == 29
