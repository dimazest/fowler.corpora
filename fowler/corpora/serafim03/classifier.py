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
from numpy.linalg import inv
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator, ClassifierMixin
from sparsesvd import sparsesvd


cache = {}


def cached_cosine(x, y):
    hash_x = hash(x.data.tobytes())
    hash_y = hash(y.data.tobytes())

    if hash_x > hash_y:
        x, y = y, x
        hash_x, hash_y = hash_y, hash_x

    try:
        return cache[hash_x, hash_y]
    except KeyError:
        r = cosine(x, y)
        cache[hash_x, hash_y] = r
        return r


class PlainLSA(BaseEstimator, ClassifierMixin):
    def __init__(self, k=100, cache=False):
        self.k = k
        self.cache = cache

    def fit(self, X, y):
        X = X.T
        self.y = y

        ut, s, vt = sparsesvd(csc_matrix(X), self.k)

        self.u = ut.T
        self.inv_s = inv(np.diag(s))
        self.v = vt.T
        self.v.flags.writeable = False

    def predict(self, X):
        u_inv_s = self.u.dot(self.inv_s)
        X_ = [x.dot(u_inv_s) for x in X]

        if self.cache:
            cosine_ = cached_cosine
        else:
            cosine_ = cosine

        def score(x_):
            for label, document in zip(self.y, self.v):
                yield cosine_(document, x_), label

        return np.array([min(score(x_))[1] for x_ in X_])
