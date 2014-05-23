import numpy as np
import pandas as pd

from fowler.corpora.dispatcher import Dispatcher, DictionaryMixin, Resource
from fowler.corpora.models import read_space_from_file, Space


class SpaceDispatcher(Dispatcher, DictionaryMixin):
    global__matrix = 'm', 'space.h5', 'Vector space.'
    global__output = 'o', 'out_space.h5', 'Output vector space file.'

    @Resource
    def space(self):
        # TODO: this is depricated, SpaceMixin should be used, and
        # global__matrix should be renamed to global__space.
        return read_space_from_file(self.matrix)


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
def pmi(
    space,
    output,
    dictionary,
    column_dictionary=('', '', 'The frequencies of column labels.'),
    column_dictionary_key=('', 'dictionary', 'An identifier for the group in the store.'),
    no_log=('', False, 'Do not take logarithm of the probability ratio.'),
    remove_missing=('', False, 'Remove items that are not in the dictionary.')
):
    """
    Weight elements using the positive PMI measure [3]. max(0, log(P(c|t) / P(c)))

    [1] and [2] use a measure similar to PMI, but without log, so it's just
    P(c|t) / P(c).

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
            row_totals = row_totals[~missing_rows]

    row_totals = row_totals.values[:, np.newaxis]

    # This is the total number of words in the corpora
    N = dictionary['count'].sum()
    # This are context probabilities in the whole Corpora P(c)
    column_totals = column_dictionary.loc[space.column_labels.index].values.flatten() / N

    # Elements in the matrix are N(c, t): the co-occurrence counts
    matrix = space.matrix.astype(float).todense()

    if remove_missing:
        matrix = matrix[~missing_rows.values]

    # The elements in the matrix are P(c|t)
    matrix /= row_totals

    if not no_log:
        # The elements in the matrix are log(P(c|t) / P(c))
        new_matrix = np.log(matrix) - np.log(column_totals)
        new_matrix[new_matrix < 0] = 0.0
    else:
        # The elements in the matrix are P(c|t) / P(c)
        new_matrix = matrix / column_totals

    Space(new_matrix, space.row_labels, space.column_labels).write(output)
