import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from fowler.corpora.space.util import write_space


class Space:
    """A vector space.

    :param data:
    :param row_lables: the row labels
    :param column_labels: the column labels

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

    def write(self, file_name):
        """Write the store to a file."""

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

    def get_target_rows(self, *labels):
        valid_labels = list(filter(None, labels))

        if valid_labels:
            vector_ids = list(filter(np.isfinite, self.row_labels.loc[valid_labels].id))
            if vector_ids:
                return self.matrix[vector_ids]

        return csr_matrix((1, self.matrix.shape[1]))

    def add(self, *targets):
        """Add vectors of the row labels (target words) element-wise."""
        vectors = self.get_target_rows(*targets)
        return csr_matrix(vectors.sum(axis=0))


def read_space_from_file(f_name):
    """Read the space form the file."""
    with pd.get_store(f_name, mode='r') as store:

        matrix = store['matrix'].reset_index()
        data = matrix['count'].values
        ij = matrix[['id_target', 'id_context']].values.T

        return Space(
            (data, ij),
            row_labels=store['targets'],
            column_labels=store['context']
        )
