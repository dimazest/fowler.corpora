import numpy as np

from fowler.corpora.serafim03.classifier import PlainLSA

import pytest


@pytest.fixture
def word_document_matrix():
    """A small word-document matrix that represents a toy dialog.

    The utterences are::

        - How are you?
        - I am fine, thank you.

        - Are you OK?
        - Yes, I am.

        - Am I OK?
        - No, you are not.

    Punctuation is ignored, the utterance tags are Q and A, for the questions
    and the answers respectively.

    Rows in the matrix correspond to the words, collumns to the documents.
    """
    return np.matrix((
        (1, 0, 0, 0, 0, 0),  # how
        (1, 0, 1, 0, 0, 1),  # are
        (1, 1, 1, 0, 0, 1),  # you
        (0, 1, 0, 1, 1, 0),  # i
        (0, 1, 0, 1, 1, 0),  # am
        (0, 1, 0, 0, 0, 0),  # fine
        (0, 1, 0, 0, 0, 0),  # thank
        (0, 0, 1, 0, 1, 0),  # ok
        (0, 0, 0, 1, 0, 0),  # yes
        (0, 0, 0, 0, 0, 1),  # no
        (0, 0, 0, 0, 0, 1),  # not
    ))


@pytest.fixture
def y():
    """The tags for the toy dialog."""
    return np.array(list('012345'))


@pytest.mark.parametrize(
    ('vector', 'expected_label'),
    (
        # How are you?
        (np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), '0'),
        # I am fine, thank you.
        (np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]), '1'),
        # I am *OK*, thank you.
        (np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0]), '1'),
    ),
)
def test_plainlsa(word_document_matrix, y, vector, expected_label):
    X = word_document_matrix.T

    cl = PlainLSA(2)

    cl.fit(X, y)

    label = cl.predict(vector)
    assert label == expected_label
