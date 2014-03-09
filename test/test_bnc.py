from fowler.corpora.bnc.util import count_cooccurrence

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
    result = count_cooccurrence(iter(sequence), window_size=window_size)

    assert sorted(result) == sorted(expected_result)
