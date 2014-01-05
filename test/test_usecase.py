from fowler.corpora.main import dispatcher

import pytest
from pytest_bdd import scenario, given, when, then


test_google_ngrams_build = scenario(
    'google_ngrams.feature',
    'Build a vector space from coocurrence matrix',
)

test_google_ngrams_tf_idf = scenario(
    'google_ngrams.feature',
    'Apply tf-id weighting to the co-occurrence matrix',
)


test_google_ngrams_line_normalize = scenario(
    'google_ngrams.feature',
    'Line-normalize the co-occurrence matrix',
)


test_google_ngrams_nmf = scenario(
    'google_ngrams.feature',
    'Reduce the co-occurrence matrix using NMF',
)


@pytest.fixture
def cooccurrence_dir_path(text_path, tmpdir):
    path_dir = tmpdir.ensure_dir('coocurrence')
    path = path_dir.join('cooccurrence_counts.csv.gz')

    dispatcher.dispatch(
        'preprocessing cooccurrence '
        '-w 5 '
        '-p {input_path} '
        '-o {output_path} '
        ''.format(
            input_path=text_path,
            output_path=path,
        ).split()
    )
    return path_dir

given('I have co-occurrence counts', fixture='cooccurrence_dir_path')


@given('I have the dictionary from the counts')
def dictionary_path(text_path, tmpdir):
    path = tmpdir.join('dictionary.h5')

    dispatcher.dispatch(
        'preprocessing dictionary '
        '-p {text_path} '
        '-o {output_path} '
        ''.format(
            text_path=text_path,
            output_path=path,
        ).split()
    )

    return path


@when('I build a co-occurrence matrix')
def build_cooccurrence_matrix(cooccurrence_dir_path, store_path, context_path, targets_path):
    dispatcher.dispatch(
        'google-ngrams cooccurrence '
        '--context {context_path} '
        '--targets {targets_path} '
        '-i {cooccurrence_dir_path} '
        '-o {output_path} '
        ''.format(
            context_path=context_path,
            cooccurrence_dir_path=cooccurrence_dir_path,
            output_path=store_path,
            targets_path=targets_path,
        ).split()
    )


@given('I select the 100 most used tokens as context')
def context_path(dictionary_path, tmpdir):
    path = tmpdir.join('contex.csv')
    dispatcher.dispatch(
        'dictionary select '
        '-d {dictionary_path} '
        '-o {context_path} '
        '--slice-end 100 '
        ''.format(
            dictionary_path=dictionary_path,
            context_path=path,
        ).split()
    )

    return path


@given('I select wordsim353 tokens as targets')
def targets_path(datadir):
    return datadir.join('targets_wordsim353.csv')


@then('I should see the evaluation report')
def i_should_see_evaluation_report():
    # import pdb; pdb.set_trace()
    pass


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


@when('I line-normalize the matrix')
def line_normalize(store_path):
    dispatcher.dispatch(
        'space line-normalize '
        '-m {input_path} '
        '-o {output_path} '
        ''.format(
            input_path=store_path,
            output_path=store_path,
        ).split()
    )


@when('I apply NMF')
def apply_nmf(store_path):
    dispatcher.dispatch(
        'space nmf '
        '-m {input_path} '
        '-o {output_path} '
        '-n 2 '
        '--tol 0.1'
        ''.format(
            input_path=store_path,
            output_path=store_path,
        ).split()
    )


@when('I evaluate the space on the wordsim353 dataset')
def evaluate_wordsim353(wordsim_353_path, store_path):
    dispatcher.dispatch(
        'wordsim353 evaluate '
        '-m {store_path} '
        '-g {wordsim_353_path} '
        ''.format(
            store_path=store_path,
            wordsim_353_path=wordsim_353_path.join('combined.csv'),
        ).split()
    )
