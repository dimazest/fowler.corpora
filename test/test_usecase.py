import pandas as pd

from fowler.corpora.main import dispatcher

import pytest
from pytest_bdd import scenario, given, when, then


test_google_ngrams_build = scenario(
    'google_ngrams.feature',
    'Build a vector space from Google Books ngrams',
)

test_google_ngrams_tf_idf = scenario(
    'google_ngrams.feature',
    'Apply tf-id weighting to the co-occurrence matrix',
)


given('I have Google Books co-occurrence counts', fixture='cooccurrence_dir_path')


@pytest.fixture
def store_path(tmpdir):
    return tmpdir.join('store.h5')


@pytest.fixture
def matrix(store_path):
    matrix = pd.read_hdf(str(store_path), 'matrix')

    assert len(matrix) == 6
    assert matrix.index.names == ['id_target', 'id_context']

    return matrix


@pytest.fixture
def counts(matrix):
    return matrix['count']


@when('I build a co-occurrence matrix')
def build_cooccurrence_matrix(cooccurrence_dir_path, store_path, context_path, targets_path):
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
            output_path=store_path,
            targets_path=targets_path,
        ).split()
    )


@then('I should have the Google co-occurrence space file')
def check_google_coocurrence_space_file(counts):
    assert counts.loc[0, 5] == 310
    assert counts.loc[0, 7] == 960
    assert counts.loc[1, 7] == 120
    assert counts.loc[2, 5] == 1300
    assert counts.loc[3, 7] == 420
    assert counts.loc[4, 6] == 510


@when('I apply tf-idf weighting')
def apply_tf_idf(store_path):
    dispatcher.dispatch(
        'space tf-idf '
        '-m {input_path} '
        '-o {output_path} '
        ''.format(
            input_path=store_path,
            output_path=store_path,
        ).split()
    )


@then('I should have the tf-idf weighted space file')
def check_tf_idf_space_file(counts):
    assert int(counts.loc[0, 5]) == 739
    assert int(counts.loc[0, 7]) == 1901
    assert int(counts.loc[1, 7]) == 237
    assert int(counts.loc[2, 5]) == 3102
    assert int(counts.loc[3, 7]) == 831
    assert int(counts.loc[4, 6]) == 1570

