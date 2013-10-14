from collections import Counter, deque
from itertools import chain, combinations, product

from opster import command, dispatch

import numpy as np
import pylab as pl
from sklearn import manifold

from composes.semantic_space.space import Space
from composes.similarity.cos import CosSimilarity

from .swda import CorpusReader


@command()
def transcripts(path):
    corpus = CorpusReader(path)

    for transcript in corpus.iter_transcripts(display_progress=False):
        for utterance in transcript.utterances:
            print('{u.caller} {u.act_tag}: {u.text}'.format(u=utterance))

        print()


def get_terms(corpus):
    for utterance in corpus.iter_utterances(display_progress=False):
        yield from utterance.pos_lemmas()  # noqa


@command()
def info(path, n=10, lemmas='True'):
    corpus = CorpusReader(path)

    terms = get_terms(corpus)
    if lemmas != 'True':
        terms = (term for term, tag in terms)

    counter = Counter(terms)

    for term, frequency in counter.most_common(int(n) or None):
        print(frequency, term)


@command()
def tags(path):
    corpus = CorpusReader(path)

    utterences = corpus.iter_utterances(display_progress=False)
    counter = Counter(u.act_tag for u in utterences)

    for tag, freq in counter.most_common():
        print(freq, tag)


def ContextBefore(utterances, n=3):
    context = deque([], n)

    for utterance in utterances:
        for c in context:
            for l, _ in c.pos_lemmas():
                yield utterance.act_tag, l

        context.append(utterance)


@command()
def cooccurrence(path, mode='inner'):
    corpus = CorpusReader(path)

    utterances = corpus.iter_utterances(display_progress=False)

    if mode == 'inner':
        pairs = chain.from_iterable(((u.act_tag, l) for l, _ in u.pos_lemmas()) for u in utterances)
    elif mode == 'before':
        pairs = ContextBefore(utterances, 3)
    else:
        raise NotImplementedError('The mode is not implemented.')

    counter = Counter(pairs)

    for (tag, lemma), count in counter.items():
        print('{} {} {}'.format(tag, lemma, count))


@command()
def space(data, rows, cols, format='sm'):
    space = Space.build(data=data, rows=rows, cols=cols, format=format)

    X = space._cooccurrence_matrix.mat

    similarities = np.zeros((X.shape[0], X.shape[0]))
    metric = CosSimilarity()
    # import ipdb; ipdb.set_trace()

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


