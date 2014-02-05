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


@when('I build a co-occurrence matrix')
def build_cooccurrence_matrix(dispatcher, cooccurrence_dir_path, store_path, context_path, targets_path):
    assert dispatcher.dispatch(
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
    ) != -1


@given('I select wordsim353 tokens as targets')
def targets_path(datadir):
    return datadir.join('targets_wordsim353.csv')


@then('I should see the evaluation report')
def i_should_see_evaluation_report():
    # Still has to be implemented...
    pass


@when('I apply tf-idf weighting')
def apply_tf_idf(dispatcher, store_path):
    assert dispatcher.dispatch(
        'space tf-idf '
        '-m {input_path} '
        '-o {output_path} '
        ''.format(
            input_path=store_path,
            output_path=store_path,
        ).split()
    ) != -1


@when('I line-normalize the matrix')
def line_normalize(dispatcher, store_path):
    assert dispatcher.dispatch(
        'space line-normalize '
        '-m {input_path} '
        '-o {output_path} '
        ''.format(
            input_path=store_path,
            output_path=store_path,
        ).split()
    ) != -1


@when('I apply NMF')
def apply_nmf(dispatcher, store_path):
    assert dispatcher.dispatch(
        'space nmf '
        '-m {input_path} '
        '-o {output_path} '
        '-n 2 '
        '--tol 0.1'
        ''.format(
            input_path=store_path,
            output_path=store_path,
        ).split()
    ) != -1


@when('I evaluate the space on the wordsim353 dataset')
def evaluate_wordsim353(dispatcher, wordsim_353_path, store_path):
    print(store_path)
    assert dispatcher.dispatch(
        'wordsim353 evaluate '
        '-m {store_path} '
        '-g {wordsim_353_path} '
        ''.format(
            store_path=store_path,
            wordsim_353_path=wordsim_353_path.join('combined.csv'),
        ).split()
    ) != -1
