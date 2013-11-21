import inspect

import opster
import pandas as pd
from jinja2 import Environment, PackageLoader

import fowler.corpora
from fowler.corpora import io


class Dispatcher(opster.Dispatcher):
    def __init__(self, *globaloptions):
        globaloptions = (
            tuple(globaloptions) +
            (
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

        path = kwargs.pop('path')

        with pd.get_store(path, mode='r') as store:
            if 'cooccurrence_matrix' in f_args:
                kwargs['cooccurrence_matrix'] = io.load_cooccurrence_matrix(store)

            if 'labels' in f_args:
                kwargs['labels'] = io.load_labels(store)

            if 'templates_env' in f_args:
                kwargs['templates_env'] = Environment(
                    loader=PackageLoader(fowler.corpora.__name__, 'templates')
                )

        return func(*args, **kwargs)

    return wrapper
