# -*- coding: utf-8 -*-
"""

.. note::

    This is taken from the scikit-learn version 0.15 and should be removed once it's released.


The :mod:`sklearn.metrics.pairwise` submodule implements utilities to evaluate
pairwise distances or affinity of sets of samples.

This module contains both distance metrics and kernels. A brief summary is
given on the two here.

Distance metrics are a function d(a, b) such that d(a, b) < d(a, c) if objects
a and b are considered "more similar" to objects a and c. Two objects exactly
alike would have a distance of zero.
One of the most popular examples is Euclidean distance.
To be a 'true' metric, it must obey the following four conditions::

    1. d(a, b) >= 0, for all a and b
    2. d(a, b) == 0, if and only if a = b, positive definiteness
    3. d(a, b) == d(b, a), symmetry
    4. d(a, c) <= d(a, b) + d(b, c), the triangle inequality

Kernels are measures of similarity, i.e. ``s(a, b) > s(a, c)``
if objects ``a`` and ``b`` are considered "more similar" to objects
``a`` and ``c``. A kernel must also be positive semi-definite.

There are a number of ways to convert between a distance metric and a
similarity measure, such as a kernel. Let D be the distance, and S be the
kernel:

    1. ``S = np.exp(-D * gamma)``, where one heuristic for choosing
       ``gamma`` is ``1 / num_features``
    2. ``S = 1. / (D / np.max(D))``
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Philippe Gervais <philippe.gervais@inria.fr>
#          Lars Buitinck <larsmans@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.utils import atleast2d_or_csr
from sklearn.utils import safe_asarray
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to (X * X).sum(axis=1), but also supports CSR sparse matrices.
    With newer NumPy versions, prevents an X.shape-sized temporary.

    Performs no input validation.
    """
    norms = (X * X).sum(axis=1)

    if not squared:
        np.sqrt(norms, norms)
    return norms


def gen_batches(n, batch_size):
    """Generator to create slices containing batch_size elements, from 0 to n.

    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.

    Examples
    --------
    >>> from sklearn.utils import gen_batches
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


# Utility Functions
def check_pairwise_arrays(X, Y):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the second dimension of the two arrays is equal.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]

    Y : {array-like, sparse matrix}, shape = [n_samples_b, n_features]

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape = [n_samples_b, n_features]
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    if Y is X or Y is None:
        X = Y = atleast2d_or_csr(X)
    else:
        X = atleast2d_or_csr(X)
        Y = atleast2d_or_csr(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    if not (X.dtype == Y.dtype == np.float32):
        if Y is X:
            X = Y = safe_asarray(X, dtype=np.float)
        else:
            X = safe_asarray(X, dtype=np.float)
            Y = safe_asarray(Y, dtype=np.float)
    return X, Y


# Distances
def pairwise_distances_argmin_min(X, Y, axis=1, metric="euclidean",
                                  batch_size=500, metric_kwargs=None):
    """Compute minimum distances between one point and a set of points.

    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance). The minimal distances are
    also returned.

    This is mostly equivalent to calling:

        (pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis),
         pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))

    but uses much less memory, and is faster for large arrays.

    Parameters
    ----------
    X, Y : {array-like, sparse matrix}
        Arrays containing points. Respective shapes (n_samples1, n_features)
        and (n_samples2, n_features)

    batch_size : integer
        To reduce memory consumption over the naive solution, data are
        processed in batches, comprising batch_size rows of X and
        batch_size rows of Y. The default value is quite conservative, but
        can be changed for fine-tuning. The larger the number, the larger the
        memory usage.

    metric : string or callable
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
        'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
        'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.

    metric_kwargs : dict, optional
        Keyword arguments to pass to specified metric function.

    Returns
    -------
    argmin : numpy.ndarray
        Y[argmin[i], :] is the row in Y that is closest to X[i, :].

    distances : numpy.ndarray
        distances[i] is the distance between the i-th row in X and the
        argmin[i]-th row in Y.

    See also
    --------
    sklearn.metrics.pairwise_distances
    sklearn.metrics.pairwise_distances_argmin
    """
    dist_func = None
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        dist_func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif not callable(metric) and not isinstance(metric, str):
        raise ValueError("'metric' must be a string or a callable")

    X, Y = check_pairwise_arrays(X, Y)

    if metric_kwargs is None:
        metric_kwargs = {}

    if axis == 0:
        X, Y = Y, X

    # Allocate output arrays
    indices = np.empty(X.shape[0], dtype=np.intp)
    values = np.empty(X.shape[0])
    values.fill(np.infty)

    for chunk_x in gen_batches(X.shape[0], batch_size):
        X_chunk = X[chunk_x, :]

        for chunk_y in gen_batches(Y.shape[0], batch_size):
            Y_chunk = Y[chunk_y, :]

            if dist_func is not None:
                if metric == 'euclidean':  # special case, for speed
                    d_chunk = safe_sparse_dot(X_chunk, Y_chunk.T,
                                                 dense_output=True)
                    d_chunk *= -2
                    d_chunk += row_norms(X_chunk, squared=True)[:, np.newaxis]
                    d_chunk += row_norms(Y_chunk, squared=True)[np.newaxis, :]
                    np.maximum(d_chunk, 0, d_chunk)
                else:
                    d_chunk = dist_func(X_chunk, Y_chunk, **metric_kwargs)

            # Update indices and minimum values using chunk
            min_indices = d_chunk.argmin(axis=1)
            min_values = d_chunk[np.arange(chunk_x.stop - chunk_x.start),
                                 min_indices]

            flags = values[chunk_x] > min_values
            indices[chunk_x] = np.where(
                flags, min_indices + chunk_y.start, indices[chunk_x])

            values[chunk_x] = np.where(
                flags, min_values, values[chunk_x])

    if metric == "euclidean" and not metric_kwargs.get("squared", False):
        np.sqrt(values, values)
    return indices, values


def pairwise_distances_argmin(X, Y, axis=1, metric="euclidean",
                              batch_size=500, metric_kwargs={}):
    """Compute minimum distances between one point and a set of points.

    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance).

    This is mostly equivalent to calling:

        pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)

    but uses much less memory, and is faster for large arrays.

    This function works with dense 2D arrays only.

    Parameters
    ==========
    X, Y : array-like
        Arrays containing points. Respective shapes (n_samples1, n_features)
        and (n_samples2, n_features)

    batch_size : integer
        To reduce memory consumption over the naive solution, data are
        processed in batches, comprising batch_size rows of X and
        batch_size rows of Y. The default value is quite conservative, but
        can be changed for fine-tuning. The larger the number, the larger the
        memory usage.

    metric : string or callable
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
        'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
        'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.

    metric_kwargs : dict
        keyword arguments to pass to specified metric function.

    Returns
    =======
    argmin : numpy.ndarray
        Y[argmin[i], :] is the row in Y that is closest to X[i, :].

    See also
    ========
    sklearn.metrics.pairwise_distances
    sklearn.metrics.pairwise_distances_argmin_min
    """
    return pairwise_distances_argmin_min(X, Y, axis, metric, batch_size,
                                         metric_kwargs)[0]


def cosine_distances(X, Y=None):
    """
    Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    distance matrix : array
        An array with shape (n_samples_X, n_samples_Y).

    See also
    --------
    sklearn.metrics.pairwise.cosine_similarity
    scipy.spatial.distance.cosine (dense matrices only)
    """
    # 1.0 - cosine_similarity(X, Y) without copy
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    return S


# Kernels
def linear_kernel(X, Y=None):
    """
    Compute the linear kernel between X and Y.

    Parameters
    ----------
    X : array of shape (n_samples_1, n_features)

    Y : array of shape (n_samples_2, n_features)

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    return safe_sparse_dot(X, Y.T, dense_output=True)


def cosine_similarity(X, Y=None):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    kernel matrix : array
        An array with shape (n_samples_X, n_samples_Y).
    """
    # to avoid recursive import

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = linear_kernel(X_normalized, Y_normalized)

    return K


# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cosine': cosine_distances,
}


def distance_metrics():
    """Valid metrics for pairwise_distances.

    This function simply returns the valid pairwise distance metrics.
    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:

    ============     ====================================
    metric           Function
    ============     ====================================
    'cosine'         metrics.pairwise.cosine_distances
    ============     ====================================

    """
    return PAIRWISE_DISTANCE_FUNCTIONS

# Helper functions - distance
PAIRWISE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cosine': cosine_similarity, }


def kernel_metrics():
    """ Valid metrics for pairwise_kernels

    This function simply returns the valid pairwise distance metrics.
    It exists, however, to allow for a verbose description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:
      ===============   ========================================
      metric            Function
      ===============   ========================================
      'additive_chi2'   sklearn.pairwise.additive_chi2_kernel
      'chi2'            sklearn.pairwise.chi2_kernel
      'linear'          sklearn.pairwise.linear_kernel
      'poly'            sklearn.pairwise.polynomial_kernel
      'polynomial'      sklearn.pairwise.polynomial_kernel
      'rbf'             sklearn.pairwise.rbf_kernel
      'sigmoid'         sklearn.pairwise.sigmoid_kernel
      'cosine'          sklearn.pairwise.cosine_similarity
      ===============   ========================================
    """
    return PAIRWISE_KERNEL_FUNCTIONS


KERNEL_PARAMS = {
    "cosine": (),
}
