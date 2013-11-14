"""Implementation of Latent Semantic Analysis for dialogue act classification.

References
----------

Serafin, Riccardo, Barbara Di Eugenio, and Michael Glass. "Latent Semantic
Analysis for dialogue act classification." Proceedings of the 2003 Conference
of the North American Chapter of the Association for Computational Linguistics
on Human Language Technology: companion volume of the Proceedings of HLT-NAACL
2003--short papers-Volume 2. Association for Computational Linguistics, 2003.

"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances


class PlainLSA(BaseEstimator, ClassifierMixin):
    """The Plain LSA method described by Serafin et al.

    :param int k: the number of dimensions the training matrix is reduced to.

    :param int n_jobs: the number of workers the distance computaion will be
    parallelized duting prediction.

    """

    def __init__(self, k=100, n_jobs=1):
        self.k = k
        self.n_jobs = n_jobs
        self.truncated_SVD = TruncatedSVD(n_components=k)

    def fit(self, X, y):
        """Fit plain LSA on the training data X.

        :param sparse matrix X: training data of the shape
        (n_samples, n_features).

        :param y: the labels of the training data.

        """
        self.y = y
        self.U_SigmaT = self.truncated_SVD.fit_transform(X)

    def predict(self, X):
        """Predict the labels of the unseen data X.

        :param sparse matrix X: training data of the shape
        (n_samples, n_features).

        :return: the predicted labels for the imput data.

        """
        X_ = self.truncated_SVD.transform(X)

        distances = pairwise_distances(
            self.U_SigmaT,
            X_,
            metric='cosine',
            n_jobs=self.n_jobs,
        )
        closest_indices = distances.argmin(axis=0)
        return self.y[closest_indices]
