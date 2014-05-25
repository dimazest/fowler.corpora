import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from fowler.corpora.space.util import write_space


logger = logging.getLogger(__name__)


class Space(Mapping):
    """A vector space.

    :param data:
    :param pandas.DataFrame row_lables: the row labels
    :param pandas.DataFrame column_labels: the column labels

    ``row_labels`` and ``column_labels`` contain at least two columns:
    ````ngram`` and id``.

    """
    def __init__(self, data_ij, row_labels, column_labels):

        self.row_labels = row_labels
        self.column_labels = column_labels

        if isinstance(data_ij, tuple):

            self.matrix = csr_matrix(
                data_ij,
                shape=(len(row_labels), len(column_labels)),
            )
        else:
            self.matrix = data_ij

        assert np.isfinite(self.matrix.data).all()

    def __iter__(self):
        return iter(self.row_labels.index)

    def __len__(self):
        return self.matrix.shape[0]

    def write(self, file_name):
        """Write the vector space to a file."""

        coo = coo_matrix(self.matrix)

        matrix = pd.DataFrame(
            {
                'count': coo.data,
                'id_target': coo.row,
                'id_context': coo.col,
            }
        ).set_index(['id_target', 'id_context'])

        write_space(file_name, self.column_labels, self.row_labels, matrix)

    def line_normalize(self):
        """Normalize the matrix, so the sum of the values in each row is 1."""
        normalized_matrix = normalize(self.matrix.astype(float), norm='l1', axis=1)

        return Space(normalized_matrix, self.row_labels, self.column_labels)

    def tf_idf(self, **kwargs):
        """Perform tf-idf transformation."""
        tfid = TfidfTransformer(**kwargs)

        tfidf_matrix = tfid.fit_transform(self.matrix)
        return Space(tfidf_matrix, self.row_labels, self.column_labels)

    def nmf(self, **kwargs):
        """Perform dimensionality reduction using NMF."""
        nmf = NMF(**kwargs)

        reduced_matrix = nmf.fit_transform(self.matrix)
        # TODO: it is incorrect to pass self.column_labels! There are not column labels.
        return Space(reduced_matrix, self.row_labels, self.column_labels)

    def __getitem__(self, key):
        """Retrive the vector for the key from the space.

        :param str key: the row label.
        :returns: a sparse matrix with one row.

        """
        vector_id, = self.row_labels.loc[key].values

        return self.matrix[vector_id]

    def get_target_rows(self, *labels, strict=False):
        """Return vectors for the labels.

        :param labels: row labels.
        :param bool strict: if `True` and a label can't be found, the KeyError exception will be risen.

        """
        valid_labels = list(filter(None, labels))

        if valid_labels:
            vector_ids = list(filter(np.isfinite, self.row_labels.loc[valid_labels].id))

            # Check that we don't return *more* vectors than asked.
            # It might be the case, that rows are POS tagged, but we query only
            # by token. It this case, two rows for `run` may be retrieved, one
            # as a verb, another as a noun.
            assert len(vector_ids) <= len(valid_labels)

            if vector_ids:
                return self.matrix[vector_ids]

        if strict:
            raise KeyError(labels)

        return csr_matrix((1, self.matrix.shape[1]))

    def add(self, *targets):
        """Add vectors of the row labels (target words) element-wise."""
        vectors = self.get_target_rows(*targets)
        return csr_matrix(vectors.sum(axis=0))

    def multiply(self, *targets):
        """Multiply vectors of the row labels (target words) element-wise."""
        vectors = self.get_target_rows(*targets)

        result = csr_matrix(np.prod(vectors.todense(), axis=0))

        assert np.isfinite(result.data).all()

        return result


def read_space_from_file(f_name):
    """Read a space form a file.

    So far, this is the preffered way to read a space.

    """
    with pd.get_store(f_name, mode='r') as store:

        matrix = store['matrix'].reset_index()
        data = matrix['count'].values

        ij = matrix[['id_target', 'id_context']].values.T

        return Space(
            (data, ij),
            row_labels=store['targets'],
            column_labels=store['context']
        )
