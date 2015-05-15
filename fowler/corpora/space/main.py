import logging

import numpy as np
import pandas as pd

from fowler.corpora.dispatcher import Dispatcher, DictionaryMixin, SpaceMixin
from fowler.corpora.models import Space


class SpaceDispatcher(Dispatcher, SpaceMixin, DictionaryMixin):
    global__output = 'o', 'out_space.h5', 'Output vector space file.'


logger = logging.getLogger(__name__)
dispatcher = SpaceDispatcher()
command = dispatcher.command


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
):
    """SVD."""
    space.svd(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state or None,
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
    times=('', ('n', 'logn'), 'Multiply the resulted values by n or logn.')
):
    """
    Weight elements using the positive PMI measure [3]. max(0, log(P(c|t) / P(c)))

    [1] and [2] use a measure similar to PMI, but without log, so it's just
    P(c|t) / P(c), which is sometimes called likelihood ratio.

    `--dictionary` provides word frequencies for rows. In case columns are
    labelled differently, provide `--column-dictionary`.

    `--keep-negative-values` keeps negative values but replaces negative
    infinity with 0. This is equivalent to replacing P(c, t) with just P(c) when
    P(c, t) is 0.

    [1] Mitchell, Jeff, and Mirella Lapata. "Vector-based Models of Semantic
    Composition." ACL. 2008.

    [2] Grefenstette, Edward, and Mehrnoosh Sadrzadeh. "Experimental support for
    a categorical compositional distributional model of meaning." Proceedings
    of the Conference on Empirical Methods in Natural Language Processing.
    Association for Computational Linguistics, 2011.

    [3] http://en.wikipedia.org/wiki/Pointwise_mutual_information

    """
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
    row_totals = dictionary.loc[space.row_labels.index]['count']

    missing_rows = ~np.isfinite(row_totals)
    if missing_rows.any():
        if not remove_missing:
            raise ValueError('These rows are not finite!', row_totals[missing_rows])
        else:
            logger.warning('Removing the following rows: %s', row_totals[missing_rows])
            row_totals = row_totals[~missing_rows]

    row_totals = row_totals.values[:, np.newaxis]

    # This are context probabilities in the whole Corpora P(c)
    column_totals = (
        column_dictionary.loc[space.column_labels.index].values.flatten()
         / dictionary['count'].sum()
    )

    # Elements in the matrix are N(c, t): the co-occurrence counts
    n = space.matrix.astype(float).todense()

    if remove_missing:
        n = n[~missing_rows.values]

    # The elements in the matrix are P(c|t)
    matrix = n / row_totals

    if not conditional_probability:
        if not no_log:
            # The elements in the matrix are log(P(c|t) / P(c))
            matrix = np.log(matrix) - np.log(column_totals)
            if keep_negative_values:
                matrix[matrix == -np.inf] = 0
            else:
                matrix[matrix < 0] = 0.0
        else:
            # The elements in the matrix are P(c|t) / P(c)
            matrix /= column_totals

    if times == 'n':
        matrix = np.multiply(n, matrix)

    Space(matrix, space.row_labels, space.column_labels).write(output)
