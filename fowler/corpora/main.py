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


@command()
def tags(path):
    corpus = CorpusReader(path)

    utterences = corpus.iter_utterances(display_progress=False)
    counter = Counter(u.act_tag for u in utterences)

    for tag, freq in counter.most_common():
        print(freq, tag)


def tokens(utterances, n=1):
    for utterance in utterances:
        ngram = deque([], n)
        for w, _ in utterance.pos_lemmas():
            ngram.append(w)
            yield utterance.act_tag, '_'.join(ngram)


def ContextBefore(utterances, context_len=3, ngram_len=1):
    context = deque([], context_len)

    for utterance in utterances:
        context.append(utterance)

        yield from tokens(context, ngram_len)


@command()
def cooccurrence(
    path,
    mode=('m', 'inner', 'Mode. innger or before'),
    context_len=('c', 3, 'Lenght of the contex in "before mode."'),
    ngram_len=('n', 1, 'Lenght of the tokens (bigrams, ngrams).')
):
    corpus = CorpusReader(path)

    utterances = corpus.iter_utterances(display_progress=False)

    if mode == 'inner':
        pairs = tokens(utterances, n=ngram_len)
    elif mode == 'before':
        pairs = ContextBefore(utterances, 3, ngram_len=ngram_len)
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


