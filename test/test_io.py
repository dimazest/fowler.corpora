import pandas as pd

from fowler.corpora.io import load_cooccurrence_matrix

from numpy.testing import assert_equal
import pytest


@pytest.yield_fixture
def store(tmpdir):
    """A store with co-occurrence counts.

    The matrix is::

        1, 7, 0, 9
        0, 0, 6, 0
        0, 0, 0, 0
        3, 0, 0, 0
        0, 9, 0, 3

    """
    with pd.get_store(str(tmpdir.join('store.hd5'))) as store:
        store['row_ids'] = pd.Series([0, 0, 0, 1, 3, 4, 4])
        store['col_ids'] = pd.Series([0, 1, 3, 2, 0, 1, 3])
        store['data'] = pd.Series([1, 7, 9, 6, 3, 9, 3])

        yield store


def test_load_cooccurrence_matrix(store):
    """Test co-cuurance matrix loading from an hd5 file."""
    matrix = load_cooccurrence_matrix(store)

    assert_equal(
        matrix.todense(),
        (
            (1, 7, 0, 9),
            (0, 0, 6, 0),
            (0, 0, 0, 0),
            (3, 0, 0, 0),
            (0, 9, 0, 3),
        ),
    )
