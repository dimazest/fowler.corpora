"""Implementation of Latent Semantic Analysis for dialogue act classification.

Usefull links
-------------

    * http://blog.josephwilk.net/projects/latent-semantic-analysis-in-python.html
      suggests to use::

        U . SIGMA' . VT = MATRIX'

    for the closest document look up.

References
----------

Serafin, Riccardo, Barbara Di Eugenio, and Michael Glass. "Latent Semantic
Analysis for dialogue act classification." Proceedings of the 2003 Conference
of the North American Chapter of the Association for Computational Linguistics
on Human Language Technology: companion volume of the Proceedings of HLT-NAACL
2003--short papers-Volume 2. Association for Computational Linguistics, 2003.

"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator, ClassifierMixin
from sparsesvd import sparsesvd


class PlainLSA(BaseEstimator, ClassifierMixin):
    def __init__(self, k=100):
        self.k = k

    def fit(self, X, y):
        X = csc_matrix(X)
        self.y = y

        ut, s, vt = sparsesvd(X, self.k)
        self.M = np.dot(ut.T, np.dot(np.diag(s), vt))

    def predict(self, X):
        _, l = min(
            (cosine(x, X), l)
            for l, x in zip(self.y, self.M)
        )
        return l
