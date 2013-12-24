import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

from fowler.corpora.dispatcher import Dispatcher

from .util import read_tokens, write_space


def middleware_hook(kwargs, f_args):
    with pd.get_store(kwargs['matrix'], mode='r') as store:

        kwargs['context'] = store['context']
        kwargs['targets'] = store['targets']
        matrix = store['matrix'].reset_index()

        counts = matrix['count'].values
        ij = matrix[['id_target', 'id_context']].values.T

        kwargs['space'] = csc_matrix(
            (counts, ij),
            shape=(len(kwargs['targets']), len(kwargs['context'])),
        )

        if 'matrix' not in f_args:
            del kwargs['matrix']


dispatcher = Dispatcher(
    middleware_hook=middleware_hook,
    globaloptions=(
        ('m', 'matrix', 'matrix.h5', 'The co-occurrence matrix.'),
    )
)
command = dispatcher.command


@command()
def select(
    context,
    targets,
    space,
    context_select=('c', '', 'The file with context words.'),
    output=('o', 'matrix_sliced.h5', 'The output matrix file.'),
):
    """Select only targets and contexts of interest."""
    if context_select:
        context_select = read_tokens(context_select)
        context = context.loc[context_select.index]

        space = space[:, context['id']]
        context['id'] = pd.Series(np.arange(len(context)), index=context.index)

    write(output, context, targets, space)


@command()
def line_normalize(
    context,
    targets,
    space,
    output=('o', 'matrix_line_normalized.h5', 'The output matrix file.'),
):
    """Normalize the matrix, so the sum of values in a row is equal to 1.

    An element $(i, j)$ in the matrix is equal to $P(c_j | t_i)$, but I'm not sure.

    """
    space = normalize(space.astype(float), norm='l1', axis=1)
    write(output, context, targets, space)


@command()
def tf_idf(
    context,
    targets,
    space,
    output=('o', 'matrix_tf-idf.h5', 'The output matrix file.'),
    norm=('', '', 'One of ‘l1’, ‘l2’ or None. Norm used to normalize term vectors. None for no normalization.'),
    use_idf=('', True, 'Enable inverse-document-frequency reweighting.'),
    smooth_idf=('', True, 'Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.'),
    sublinear_tf=('', False, 'Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).'),
):
    """Perform tf-idf transformation."""
    tfid = TfidfTransformer(
        norm=norm or None,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )
    space = tfid.fit_transform(space)

    write(output, context, targets, space)


def write(output, context, targets, space):
    space = space.tocoo()

    matrix = pd.DataFrame(
        {
            'count': space.data,
            'id_target': space.row,
            'id_context': space.col,
        }
    )

    write_space(output, context, targets, matrix)
