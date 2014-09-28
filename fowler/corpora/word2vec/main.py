"""Helpers to access word2vec vectors."""
import pandas as pd
from gensim.models import Word2Vec

from fowler.corpora.dispatcher import Dispatcher, NewSpaceCreationMixin
from fowler.corpora.models import Space


class Word2VecDispathcer(Dispatcher, NewSpaceCreationMixin):
    """Word2vec dispatcher."""


dispatcher = Word2VecDispathcer()
command = dispatcher.command


@command()
def to_space(
    word2vec=('', 'GoogleNews-vectors-negative300.bin.gz', 'Path to word2vec vectors.'),
    output=('o', 'space.h5', 'The output space file.'),
):
    """Read a word2vec file and save it as a space file."""
    model = Word2Vec.load_word2vec_format(word2vec, binary=True)

    targets = pd.DataFrame({'id': range(len(model.index2word))}, index=model.index2word)
    targets.index.name = 'ngram'

    context = pd.DataFrame({'id': range(model.syn0.shape[1])})
    context.index.name = 'ngram'

    space = Space(
        data_ij=model.syn0,
        row_labels=targets,
        column_labels=context,
    )

    space.write(output)
