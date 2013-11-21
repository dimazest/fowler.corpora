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
from sklearn.metrics import pairwise_distances_argmin


class PlainLSA(BaseEstimator, ClassifierMixin):
    """The Plain LSA method described by Serafin et al.

    :param int k: the number of dimensions in the reduced vectors.

    """

    def __init__(self, k=100):
        self.k = k

    def fit(self, X, y):
        """Train the model on the labeled documents X.

        :param sparse matrix X: training data of the shape
        (n_samples, n_features).

        :param y: the labels of the training data.

        Perform the SVD decomposition. The documents transformed to the reduced
        vector space are stored in `U_SigmaT`. The SVD model itself is stored
        in `truncated_SVD`.

        """
        self.y = y
        self.truncated_SVD = TruncatedSVD(n_components=self.k)
        self.U_SigmaT = self.truncated_SVD.fit_transform(X)

    def predict(self, X):
        """Predict the labels of the unlabeled documents X.

        :param sparse matrix X: unlabeled data of the shape
        (n_samples, n_features).

        :return: the labels for the input data.

        The input documents are transformed into the reduced vector space. Then
        for each transformed vector the closest vector in the training data is
        looked up. The label of that document is the label of the input
        document.

        """

        X_ = self.truncated_SVD.transform(X)
        closest_indices = pairwise_distances_argmin(
            X_,
            self.U_SigmaT,
            metric='cosine',
        )

        return self.y[closest_indices]
