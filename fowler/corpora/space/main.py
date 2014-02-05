from fowler.corpora import dispatcher
from fowler.corpora.models import read_space_from_file


class Dispatcher(dispatcher.Dispatcher):
    global__matrix = 'm', 'matrix.h5', 'The co-occurrence matrix.'
    global__output = 'o', 'out_space.h5', 'The output matrix file.'

    @dispatcher.Resource
    def space(self):
        return read_space_from_file(self.matrix)


dispatcher = Dispatcher()
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
