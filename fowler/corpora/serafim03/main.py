"""Implementation of Latent Semantic Analysis for dialogue act classification."""
import numpy as np

from fowler.corpora.main.options import Dispatcher

from .classifier import PlainLSA

dispatcher = Dispatcher()
command = dispatcher.command


@command()
def plain_lsa(
    cooccurrence_matrix,
    k=('k', 100, 'The number of dimensions after SVD applicaion.'),
):
    """Perform the Plain LSA method."""
    X = cooccurrence_matrix
    y = np.zeroes(len(X))

    c = PlainLSA(k)
    c.fit(X, y)

    import ipdb; ipdb.set_trace()


