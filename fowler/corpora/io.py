"""IO functions."""
import fileinput
import contextlib
import codecs

import numpy as np
from scipy.sparse import csc_matrix

from py.path import local


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


def load_labels(store):
    return store['labels'].values.astype(str)


@contextlib.contextmanager
def readline_folder(input_dir=None):
    """Open all files in the `input_dir` as one big file.

    Uncompresses the files on the fly if needed.

    """
    file_names = local(input_dir).listdir() if input_dir else []
    file_names = map(str, file_names)

    with contextlib.closing(
        fileinput.FileInput(
            file_names,
            openhook=fileinput.hook_compressed,
            mode='rb',
        ),
    ) as lines:
        yield codecs.iterdecode(lines, 'utf-8')
