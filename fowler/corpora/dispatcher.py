import inspect
import logging
from multiprocessing import Pool

import pandas as pd

import opster
from jinja2 import Environment, PackageLoader

import fowler.corpora


class Dispatcher(opster.Dispatcher):
    def __init__(self, globaloptions=tuple(), middleware_hook=None):
        globaloptions = (
            tuple(globaloptions) +
            (
                ('v', 'verbose', False, 'Be verbose.'),
                ('j', 'jobs_num', 0, 'Number of jobs for parallel tasks.'),
                ('', 'display_max_rows', 0, 'Maximum number of rows to show in pandas.'),
            )
        )

        self.middleware_hook = middleware_hook

        super(Dispatcher, self).__init__(
            globaloptions=globaloptions,
            middleware=self._middleware,
        )

    def _middleware(self, func):
        def wrapper(*args, **kwargs):
            if func.__name__ == 'help_inner':
                return func(*args, **kwargs)

            f_args = inspect.getargspec(func)[0]

            display_max_rows = kwargs.pop('display_max_rows')
            if display_max_rows:
                pd.set_option('display.max_rows', display_max_rows)

            verbose = kwargs['verbose']

            logging.captureWarnings(True)
            logger = logging.getLogger('fowler')
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)-6s: %(name)s - %(levelname)s - %(message)s')

            handler.setFormatter(formatter)
            logger.addHandler(handler)

            if verbose:
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(logging.CRITICAL)

            if self.middleware_hook:
                self.middleware_hook(kwargs, f_args)

            # Remove global options if we don't need them.
            if 'verbose' not in f_args:
                del kwargs['verbose']

            if 'pool' in f_args:
                kwargs['pool'] = Pool(kwargs['jobs_num'] or None)

            if 'jobs_num' not in f_args:
                del kwargs['jobs_num']

            if 'templates_env' in f_args:
                kwargs['templates_env'] = Environment(
                    loader=PackageLoader(fowler.corpora.__name__, 'templates')
                )

            kwarg_keys = sorted(kwargs.keys())
            sorted_fargs = sorted(f_args)
            if kwarg_keys != sorted_fargs:
                logger.debug('Key mismatch. kwargs: %s. fargs %s ', kwarg_keys, sorted_fargs)

            func(*args, **kwargs)

        return wrapper
