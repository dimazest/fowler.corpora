"""IO functions."""

import numpy as np
from scipy.sparse import csc_matrix


def load_cooccurrence_matrix(store, matrix_type=csc_matrix):
    """Load a co-occurrence matrix from a store."""

    ij = np.vstack((
        store['row_ids'].values,
        store['col_ids'].values,
    ))

    matrix = matrix_type((
        store['data'].values,
        ij,
    ))

    return matrix
