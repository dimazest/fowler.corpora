import logging

import numpy as np
import pandas as pd

from scipy import sparse

from fowler.corpora.dispatcher import Dispatcher, DictionaryMixin, SpaceMixin
from fowler.corpora.models import Space, read_space_from_file


class SpaceDispatcher(Dispatcher, SpaceMixin, DictionaryMixin):
    global__output = 'o', 'out_space.h5', 'Output vector space file.'


logger = logging.getLogger(__name__)
dispatcher = SpaceDispatcher()
command = dispatcher.command


@command()
def truncate(
    space,
    output,
    size=('', 2000, 'New vector length.'),
    nvaa=('', False, 'Use only nouns, verbs, adjectives and adverbs as features.'),
    tagset=('', '', 'Tagset'),
):
    assert space.matrix.shape[1] >= size

    features = space.column_labels
    if nvaa:
        if tagset == 'bnc':
            features = features[features.index.get_level_values('tag').isin(['SUBST', 'VERB', 'ADJ', 'ADV'])]
        else:
            features = features[features.index.get_level_values('tag').isin(['N', 'V', 'J', 'R'])]

    # It's important to sort by id to make sure that the most frequent features are selected.
    features = features.sort('id').head(size)
    matrix = sparse.csc_matrix(space.matrix)[:, features['id']]

    assert len(features) == size

    # Reindex features
    features['id'] = list(range(size))

    new_space = Space(
        matrix,
        row_labels=space.row_labels,
        column_labels=features,
    )

    new_space.write(output)


@command()
def line_normalize(space, output):
    """Normalize the matrix, so the sum of values in a row is equal to 1."""
    space.line_normalize().write(output)


@command()
def log(space, output):
    """Logarithms."""
    space.log().write(output)


@command()
def tf_idf(
    space,
    output,
    norm=('', '', 'One of ‘l1’, ‘l2’ or None. Norm used to normalize term vectors. None for no normalization.'),
    use_idf=('', True, 'Enable inverse-document-frequency reweighting.'),
    smooth_idf=('', False, 'Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.'),
    sublinear_tf=('', False, 'Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).'),
):
    """Perform tf-idf transformation."""
    space.tf_idf(
        norm=norm or None,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    ).write(output)


@command()
def nmf(
    space,
    output,
    n_components=('n', 100, 'Number of components.'),
    init=('', '', 'Method used to initialize the procedure. [nndsvd|nndsvda|nndsvdar|random]'),
    sparseness=('', '', 'Where to enforce sparsity in the model. [data|components]'),
    beta=('', 1.0, 'Degree of sparseness. Larger values mean more sparseness.'),
    eta=('', 0.1, 'Degree of correctness to maintain, if sparsity is not None. Smaller values mean larger error.'),
    tol=('', 1e-4, 'Tolerance value used in stopping conditions.'),
    max_iter=('', 200, 'Number of iterations to compute.'),
    nls_max_iter=('', 2000, 'Number of iterations in NLS subproblem.'),
    random_state=('', 0, 'Random number generator seed control. 0 for undefined.'),
):
    """Non-Negative matrix factorization.

    :param str init: Method used to initialize the procedure.
        Default: 'nndsvdar' if n_components < n_features, otherwise random.

        Valid options:
        * 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
        initialization (better for sparseness)

        * 'nndsvda': NNDSVD with zeros filled with the average of X
        (better when sparsity is not desired)

        * 'nndsvdar': NNDSVD with zeros filled with small random values
        (generally faster, less accurate alternative to NNDSVDa
        for when sparsity is not desired)

        * 'random': non-negative random matrices

    """
    space.nmf(
        n_components=n_components,
        init=init or None,
        sparseness=sparseness or None,
        beta=beta,
        eta=eta,
        tol=tol,
        max_iter=max_iter,
        nls_max_iter=nls_max_iter,
        random_state=random_state or None,
    ).write(output)


@command()
def svd(
    space,
    output,
    n_components=('n', 300, 'Number of components.'),
    n_iter=('', 5, 'Number of iterations for randomized SVD solver.'),
    random_state=('', 0, 'Random number generator seed control. 0 for undefined.'),
    l2_normalize=('', False, 'L2-normalize vectors before decomposition.'),
):
    """SVD."""
    space.svd(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state or None,
        l2_normalize=l2_normalize,
    ).write(output)


@command()
def pmi(
    space,
    output,
    dictionary,
    column_dictionary=('', '', 'The frequencies of column labels.'),
    column_dictionary_key=('', 'dictionary', 'An identifier for the group in the store.'),
    no_log=('', False, 'Do not take logarithm of the probability ratio.'),
    remove_missing=('', False, 'Remove items that are not in the dictionary.'),
    conditional_probability=('', False, 'Compute only P(c|t).'),
    keep_negative_values=('', False, 'Keep negative values.'),
    neg=('', 1.0, 'The K parameter for shifted PPMI.'),
    log_base=('', np.e, 'The logarithm base to use.'),
    times=('', ('', 'n', 'logn'), 'Multiply the resulted values by n or log(n+1).'),
    window_size=('', 10, 'The size of the window.'),
    cds=('', float('nan'), 'Context discounting smoothing cooficient.'),
    smoothing=('', ('minprob', 'chance', 'compress'), 'How to deal with unseen co-occurrence prbailty.'),
):
    """
    Weight elements using the positive PMI measure [3]. max(0, log(P(c|t) / P(c)))

    [1] and [2] use a measure similar to PMI, but without log, so it's just
    P(c|t) / P(c), which is sometimes called likelihood ratio.

    `--dictionary` provides word frequencies for rows. In case columns are
    labelled differently, provide `--column-dictionary`.

    [1] Mitchell, Jeff, and Mirella Lapata. "Vector-based Models of Semantic
    Composition." ACL. 2008.

    [2] Grefenstette, Edward, and Mehrnoosh Sadrzadeh. "Experimental support for
    a categorical compositional distributional model of meaning." Proceedings
    of the Conference on Empirical Methods in Natural Language Processing.
    Association for Computational Linguistics, 2011.

    [3] http://en.wikipedia.org/wiki/Pointwise_mutual_information

    """

    if log_base == np.e:
        log = np.log
        log1p = np.log1p
    else:
        def log(x, out=None):
            result = np.log(x, out)
            result /= np.log(log_base)

            return result

        def log1p(x, out=None):
            result = np.log1p(x, out)
            result /= np.log(log_base)

            return result

    def set_index(dictionary):
        dictionary.set_index(
            [c for c in dictionary.columns if c != 'count'],
            inplace=True,
        )

    set_index(dictionary)

    if column_dictionary:
        column_dictionary = pd.read_hdf(column_dictionary, key=column_dictionary_key)
        set_index(column_dictionary)
    else:
        column_dictionary = dictionary

    # This are target frequency counts in the whole Corpora N(t)
    row_totals = dictionary.loc[space.row_labels.sort('id').index]['count']

    missing_rows = ~np.isfinite(row_totals)
    if missing_rows.any():
        if not remove_missing:
            raise ValueError('These rows are not finite!', row_totals[missing_rows])
        else:
            logger.warning('Not finite rows: %s', row_totals[missing_rows])

    N = dictionary['count'].sum()

    row_totals[missing_rows] = 1
    row_totals = row_totals.values[:, np.newaxis] / N

    if np.isnan(cds):
        # Use dictionary for context total counts.
        column_totals = (
            column_dictionary.loc[space.column_labels.index].values.flatten() / N
        )
    else:
        # Use co-occurrence matrix for context co-occurrence counts.

        # Prepare for the Context Distribution Smoothing.
        smoothed_context_counts = np.array(space.matrix.sum(axis=0)).flatten() ** cds

        # This are context probabilities in the whole Corpora P(c)
        column_totals = smoothed_context_counts / smoothed_context_counts.sum()

    # Elements in the matrix are N(c, t): the co-occurrence counts
    n = space.matrix.astype(float).todense()

    # The elements in the matrix are P(c, t)
    matrix = n / (N * window_size)

    matrix_sum = matrix.sum()
    assert matrix_sum < 1.0 or np.isclose(matrix_sum, 1.0)

    # # Check that P(c|t) <= 1.
    # max_row_sum = (matrix / (column_totals * row_totals)).sum(axis=1).max()
    # assert max_row_sum < 1.0 or np.isclose(max_row_sum, 1.0)

    if not conditional_probability:
        if not no_log:
            # PMI
            zero_counts = matrix == 0

            if smoothing == 'minprob':
                # Pretned that unseen pairs occurred once.
                matrix[zero_counts] = 1 / (N * window_size)

            if smoothing != 'compress':
                # The elements in the matrix are log(P(c, t))
                log(matrix, matrix)

                # log(P(c, t)) - (log(P(c)) + log(P(t)))
                matrix -= log(column_totals)
                matrix -= log(row_totals)
            else:
                matrix /= column_totals * row_totals
                matrix = log1p(matrix, matrix)

            if smoothing in ('chance', 'compress'):
                matrix[zero_counts] = 0

            if not keep_negative_values:
                # PPMI
                if smoothing == 'compress':
                    matrix -= log(2)

                if neg != 1.0:
                    matrix -= log(neg)

                matrix[matrix < 0] = 0.0

        else:
            # Ratio
            # The elements in the matrix are P(c,t) / ((P(c) * P(t)))
            matrix /= column_totals * row_totals
    else:
        # Conditional: P(c|t)
        matrix /= row_totals
        max_row_sum = (matrix).sum(axis=1).max()
        assert max_row_sum < 1.0 or np.isclose(max_row_sum, 1.0)

    if times == 'n':
        matrix = np.multiply(n, matrix)
    if times == 'logn':
        matrix = np.multiply(np.log(n + 1), matrix)

    Space(matrix, space.row_labels, space.column_labels).write(output)


@command()
def ittf(
    space,
    output,
    dictionary,
    raw_space=('', '', 'Space with feature co-occurrence counts.'),
    times=('', ('n', 'logn'), 'Multiply the resulted values by n or logn.'),
):
    raw_space = read_space_from_file(raw_space)

    feature_cardinality = np.array(
        [v.nnz for v in raw_space.get_target_rows(*space.column_labels.index)]
    )

    n = space.matrix.todense()

    ittf = np.log(len(dictionary)) - np.log(feature_cardinality)

    if times == 'n':
        matrix = np.multiply(n, ittf)
    elif times == 'logn':
        matrix = np.multiply(np.log(n + 1), ittf)

    Space(matrix, space.row_labels, space.column_labels).write(output)
