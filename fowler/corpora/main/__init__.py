from itertools import product

import numpy as np
import pylab as pl
from sklearn import manifold

from composes.semantic_space.space import Space
from composes.similarity.cos import CosSimilarity

from .options import Dispatcher
from fowler.corpora.serafim03 import main as serafim03_main

dispatcher = Dispatcher()
command = dispatcher.command
dispatch = dispatcher.dispatch


dispatcher.nest(
    'serafin03',
    serafim03_main.dispatcher,
    serafim03_main.__doc__,
)


@command()
def info(cooccurrence_matrix):
    print(
        'The co-coocurance matrix shape is {m.shape}.'
        ''.format(m=cooccurrence_matrix)
        )


@command()
def draw_space(data, rows, cols, format='sm'):
    space = Space.build(data=data, rows=rows, cols=cols, format=format)

    X = space._cooccurrence_matrix.mat

    similarities = np.zeros((X.shape[0], X.shape[0]))
    metric = CosSimilarity()

    for (i, li), (j, lj) in product(list(enumerate(space.id2row)), repeat=2):
        similarities[i, j] = 1 - space.get_sim(li, lj, similarity=metric)

    clf = manifold.MDS(
        n_components=2,
        n_init=1,
        max_iter=100,
        dissimilarity='precomputed',
        n_jobs=-1,
    )
    X_mds = clf.fit_transform(similarities)

    plot_embedding(X_mds, space, '{} {}'.format(data, rows))

    pl.savefig('{}_{}.pdf'.format(data, rows))


def plot_embedding(X, space, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    pl.figure()

    colors = {l[0]: c for c, l in enumerate(space.id2row)}

    for i in range(X.shape[0]):
        label = space.id2row[i]

        pl.text(
            X[i, 0],
            X[i, 1],
            label,
            color=pl.cm.Set1(colors[label[0]]),
            fontdict={'weight': 'bold', 'size': 9}
        )

    if title is not None:
        pl.title(title)
