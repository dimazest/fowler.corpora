import inspect

import opster
import pandas as pd

from .io import load_cooccurrence_matrix


class Dispatcher(opster.Dispatcher):
    def __init__(self, *globaloptions):
        globaloptions = (
            tuple(globaloptions) +
            (
                ('v', 'verbose', False, 'Be verbose.'),
                ('p', 'path', 'out.h5', 'The path to the store hd5 file.'),
            )
        )

        super(Dispatcher, self).__init__(
            globaloptions=globaloptions,
            middleware=_middleware,
        )


def _middleware(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == 'help_inner':
            return func(*args, **kwargs)

        f_args = inspect.getargspec(func)[0]

        verbose = kwargs.pop('verbose')
        path = kwargs.pop('path')

        if 'cooccurrence_matrix' in f_args:
            with pd.get_store(path, mode='r') as store:
                cooccurrence_matrix = load_cooccurrence_matrix(store)
            kwargs['cooccurrence_matrix'] = cooccurrence_matrix

        return func(*args, **kwargs)

    return wrapper
