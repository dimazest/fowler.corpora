from itertools import chain
from collections import Counter

import pandas as pd

from fowler.corpora.bnc.util import co_occurrences
from fowler.corpora.dispatcher import DictionaryMixin
from fowler.corpora.categorical.main import CategoricalDispatcher

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
        (
            'abbc',
            (1, 2),
            [
                ('a', 'b', 2),
                ('b', 'a', 1),
                ('b', 'b', 2),
                ('b', 'c', 2),
                ('c', 'b', 1),
            ],
        ),
        (
            'abc',
            (1, 0),
            [
                ('b', 'a', 1),
                ('c', 'b', 1),
            ],
        ),
    ),
)
def test_count_cooccurrence(sequence, window_size, expected_result):
    try:
        window_size_before, window_size_after = window_size
    except TypeError:
        window_size_before = window_size_after = window_size

    result = co_occurrences(sequence, window_size_before=window_size_before, window_size_after=window_size_after)
    result = Counter(
        chain.from_iterable(
            ((t,  c) for c in cs) for t, cs in result
        )
    )

    expected_result = Counter({(c, t): n for c, t, n in expected_result})

    assert result == expected_result

@pytest.mark.parametrize(
    ('stem', 'tag_first_letter', 'ngrams', 'counts', 'dictionary_len'),
    (
        ('', '', [('.', 'PUN'), ('she', 'PRON'), (',', 'PUN'), ('to', 'PREP'), ('and', 'CONJ')], [11, 9, 8, 7, 5], 88),
        ('--stem', '', [('.', 'PUN'), ('she', 'PRON'), (',', 'PUN'), ('to', 'PREP'), ('and', 'CONJ')], [11, 15, 8, 7, 6], 73),
        ('', '--tag_first_letter', [('.', 'P'), ('she', 'P'), (',', 'P'), ('to', 'P'), ('and', 'C')], [11, 9, 8, 7, 5], 88),
        ('--stem', '--tag_first_letter', [('.', 'P'), ('she', 'P'), (',', 'P'), ('to', 'P'), ('and', 'C')], [11, 15, 8, 7, 6], 73),
    )
)
def test_bnc_dictionary(bnc_path, dispatcher, tmpdir, stem, tag_first_letter, ngrams, counts, dictionary_len):
    dictionary_path = str(tmpdir.join("dictionary.h5"))
    dispatcher.dispatch(
        'bnc dictionary '
        '--corpus bnc://{corpus} '
        '-o {output} '
        '--no_p11n '
        '{tag_first_letter} '
        ''.format(
            corpus=bnc_path,
            output=dictionary_path,
            stem='stem=true&' if stem else '',
            tag_first_letter='tag_first_letter=true&' if tag_first_letter else '',
        ).split()
    )

    dictionary = DictionaryMixin.get_dictionary(dictionary_path, 'dictionary')

    dictionary.set_index(['ngram', 'tag'], inplace=True)
    assert (dictionary.loc[ngrams]['count'] == counts).all()

    # Extra counts are because of added ('', '') tokens before and after the sentences.
    assert dictionary['count'].sum() == 151 + dictionary.loc[('', ''), 'count']
    assert len(dictionary) == dictionary_len + 1


@pytest.mark.parametrize(
    ('stem', 'tag_first_letter', 'ngrams', 'counts', 'dictionary_len'),
    (
        (False, False, [('.', '.'), ('she', 'PRP'), (',', ','), ('to', 'TO'), ('and', 'CC')], [117, 1, 55, 39, 22], 675),
        (True, False, [('.', '.'), ('she', 'PRP'), (',', ','), ('to', 'TO'), ('and', 'CC')], [117, 1, 55, 39, 24], 621),
        (False, True, [('.', '.'), ('she', 'P'), (',', ','), ('to', 'T'), ('and', 'C')], [117, 1, 55, 39, 22], 654),
        (True, True, [('.', '.'), ('she', 'P'), (',', ','), ('to', 'T'), ('and', 'C')], [117, 1, 55, 39, 24], 547),
    )
)
def test_bnc_ccg_dictionary(bnc_ccg_path, dispatcher, tmpdir, stem, tag_first_letter, ngrams, counts, dictionary_len):
    dictionary_path = str(tmpdir.join("dictionary.h5"))
    dispatcher.dispatch(
        'bnc dictionary '
        '--corpus bnc+ccg://{corpus}?{stem}{tag_first_letter} '
        '-o {output} '
        '--no_p11n '
        ''.format(
            corpus=bnc_ccg_path,
            output=dictionary_path,
            stem='stem=true&' if stem else '',
            tag_first_letter='tag_first_letter=true&' if tag_first_letter else '',
        ).split()
    )

    dictionary = DictionaryMixin.get_dictionary(dictionary_path, 'dictionary')

    dictionary.set_index(['ngram', 'tag'], inplace=True)

    assert (dictionary.loc[ngrams]['count'] == counts).all()

    # Extra counts are because of added ('', '') tokens before and after the sentences.
    assert dictionary['count'].sum() == 1781 + dictionary.loc[('', ''), 'count']
    assert len(dictionary) == dictionary_len + 1


def test_bnc_cooccurrence(space):
    assert space.matrix.sum() == 29


def test_bnc_ccg_transitive_verbs(bnc_ccg_path, dispatcher, tmpdir):
    vso_path = str(tmpdir.join("dictionary.h5"))
    dispatcher.dispatch(
        'bnc transitive-verbs '
        '--corpus bnc+ccg://{corpus} '
        '-o {output} '
        '--no_p11n '
        ''.format(
            corpus=bnc_ccg_path,
            output=vso_path,
        ).split()
    )

    vso = CategoricalDispatcher.read_vso_file(vso_path, 'dictionary')

    vso.set_index(
        ['verb', 'verb_stem', 'verb_tag', 'subj', 'subj_stem', 'subj_tag', 'obj', 'obj_stem', 'obj_tag'],
        inplace=True,
    )
    assert len(vso) == 74


def test_bnc_ccg_dependencies(bnc_ccg_path, dispatcher, tmpdir):
    path = str(tmpdir.join("dependencies.h5"))
    dispatcher.dispatch(
        'bnc dependencies '
        '--corpus bnc+ccg://{corpus} '
        '-o {output} '
        '--no_p11n '
        ''.format(
            corpus=bnc_ccg_path,
            output=path,
        ).split()
    )

    result = pd.read_hdf(path, 'dictionary')

    assert result.index.is_unique
    assert len(result) == 1267
    assert result['count'].sum() == 1413

    assert len(set(result.index.levels[2])) == len(result.index.levels[2])

    assert set(result.index.levels[2]) == {
        'aux',
        'ccomp',
        'cmod',
        'conj',
        'det',
        'dobj',
        'iobj',
        'ncmod',
        'ncsubj',
        'obj2',
        'xcomp',
        'xmod',
    }
