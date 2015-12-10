"""Helpers to access word2vec vectors."""
import pandas as pd

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
    word2vec_format=('', False, 'Word2vec_format.'),
    pos_separator=('', '', 'POS separator.'),
):
    """Read a word2vec file and save it as a space file."""
    from gensim.models import Word2Vec

    if word2vec_format:
        model = Word2Vec.load_word2vec_format(word2vec, binary=True)
    else:
        model = Word2Vec.load(word2vec)

    if not pos_separator:
        targets = pd.DataFrame(
            {
                'id': range(len(model.index2word)),
                'ngram': model.index2word,
                'tag': '_',
            },
        )
    else:
        tokens = [s.rsplit(pos_separator, maxsplit=1) for s in model.index2word]
        targets = pd.DataFrame(
            {
                'id': range(len(model.index2word)),
                'ngram': [n for n, _ in tokens],
                'tag': [t for _, t in tokens],
            },
        )

    targets.set_index(['ngram', 'tag'], inplace=True)

    context = pd.DataFrame(
        {
            'id': range(model.syn0.shape[1]),
            'ngram': range(model.syn0.shape[1]),
            'tag': '_'
        },

    )
    context.set_index(['ngram', 'tag'], inplace=True)

    space = Space(
        data_ij=model.syn0,
        row_labels=targets,
        column_labels=context,
    )

    space.write(output)
