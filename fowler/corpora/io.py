"""IO functions."""

import numpy as np
from scipy.sparse import csr_matrix


def load_cooccurrence_matrix(store):
    """Load a co-occurrence matrix from a store."""

    ij = np.vstack((
        store['row_ids'].values,
        store['col_ids'].values,
    ))

    matrix = csr_matrix((
        store['data'].values,
        ij,
    ))

    return matrix
