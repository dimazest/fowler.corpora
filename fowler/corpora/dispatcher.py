import inspect
import logging
import os

from logging.handlers import RotatingFileHandler
from multiprocessing import Pool

import numpy as np
import pandas as pd

import raven
from raven.conf import setup_logging
from raven.handlers.logging import SentryHandler

import opster

from jinja2 import Environment, PackageLoader
from zope.cachedescriptors.property import Lazy

import joblib
from IPython import parallel

import fowler.corpora
from fowler.corpora.space.util import read_tokens
from fowler.corpora.models import read_space_from_file


class Resource(Lazy):
    """A resource."""


class EagerResource(Resource):
    """A resource that is evaluated even if it's not requested."""


class BaseDispatcher(opster.Dispatcher):
    """Base dispathcer with generic basic resources."""

    def __init__(self):
        global_option_prefix = 'global__'
        global_names = [g for g in dir(self) if g.startswith(global_option_prefix)]
        global_params = {g[len(global_option_prefix):]: getattr(self, g) for g in global_names}
        globaloptions = [(short, name, default, help) for name, (short, default, help) in global_params.items()]

        super().__init__(
            globaloptions=globaloptions,
            middleware=self._middleware,
        )

    def _middleware(self, func):
        def wrapper(*args, **kwargs):
            if func.__name__ == 'help_inner':
                return func(*args, **kwargs)

            argspec = inspect.getfullargspec(func)
            f_args = argspec.args

            # It's a dirty hack...
            self.kwargs = kwargs

            # Look for eager resources
            t = type(self)
            for eager_resource in (r for r in dir(t) if isinstance(getattr(t, r), EagerResource)):
                getattr(self, eager_resource)

            f_kwargs = {f_arg: getattr(self, f_arg) for f_arg in f_args}

            return func(**f_kwargs)

        return wrapper

    def __getattr__(self, name):
        return self.kwargs[name]


class DummyPool:

    def imap_unordered(self, func, iterable, chunksize=None):
        return self.map(func, iterable)

    def map(self, func, iterable, chunksize=None):
        return list(map(func, iterable))


class Dispatcher(BaseDispatcher):
    global__display_max_rows = '', 0, 'Maximum number of rows to show in pandas.'
    global__job_num = 'j', 0, 'Number of jobs for parallel tasks.'
    global__limit = '', 0, 'Number of elements to limit.'
    global__logger_backup_count = '', 1000, 'The number of log messages to keep.'
    global__logger_filename = '', '/tmp/fowler.log', 'File to log.'
    global__no_p11n = '', False, "Don't parrallelize the code across several "
    global__verbose = 'v', False, 'Be verbose.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @Resource
    def pool(self):
        if self.no_p11n:
            return DummyPool()

        return Pool(self.job_num or None)

    @Resource
    def templates_env(self):
        return Environment(
            loader=PackageLoader(fowler.corpora.__name__, 'templates')
        )

    @EagerResource
    def logger(self):
        logging.captureWarnings(True)
        logger = logging.getLogger()
        handler = RotatingFileHandler(
            filename=self.logger_filename,
            backupCount=self.logger_backup_count,
        )
        formatter = logging.Formatter('%(asctime)-6s: %(name)s - %(levelname)s - %(process)d - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if self.sentry_handler:
            logger.addHandler(self.sentry_handler)

        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.CRITICAL)

        return logger

    @EagerResource
    def display_max_rows(self):
        display_max_rows = self.kwargs['display_max_rows']
        if display_max_rows:
            pd.set_option('display.max_rows', display_max_rows)

        return display_max_rows

    @Resource
    def ip_client(self):
        """IPython parallel infrastructure client."""
        return parallel.Client()

    @Resource
    def ip_view(self):
        """IPython's parallel cluster view object."""
        return self.ip_client[:]

    @Resource
    def parallel(self):
        """Joblib parallel pool."""
        return joblib.Parallel(n_jobs=self.job_num)

    @Resource
    def sentry_client(self):
        """Setntry client."""
        if 'SENTRY_DSN' in os.environ:
            return raven.Client()

    @Resource
    def sentry_handler(self):
        """Sentry log handler."""
        if self.sentry_client:
            handler = SentryHandler(self.sentry_client)
            setup_logging(handler)
            return handler


class SpaceCreationMixin:
    """A mixin that defines common arguments for space creation commands."""

    global__context = 'c', 'context.csv', 'The file with context words.'
    global__targets = 't', 'targets.csv', 'The file with target words.'

    @Resource
    def targets(self):
        context = read_tokens(self.kwargs['context'])
        context['id'] = pd.Series(np.arange(len(context)), index=context.index)

        return context

    @Resource
    def context(self):
        targets = read_tokens(self.kwargs['targets'])
        targets['id'] = pd.Series(np.arange(len(targets)), index=targets.index)

        return targets


class NewSpaceCreationMixin:
    """A mixin that defines common arguments for space creation commands.

    A token file have to have a header and may have several columns. The only
    obligatory is ``ngram``.

    """

    global__context = 'c', 'context.csv', 'The file with context words.'
    global__targets = 't', 'targets.csv', 'The file with target words.'

    def read_file(self, f_name):
        frame = pd.read_csv(f_name, encoding='utf8')

        frame.set_index(frame.columns.tolist(), inplace=True)
        frame['id'] = pd.Series(np.arange(len(frame)), index=frame.index)

        assert frame.index.is_unique

        return frame

    @Resource
    def targets(self):
        return self.read_file(self.kwargs['targets'])

    @Resource
    def context(self):
        return self.read_file(self.kwargs['context'])


class SpaceMixin:
    """A mixin that provides access to the space object."""
    global__space = 's', 'space.h5', 'The vector space.'

    @Resource
    def space(self):
        return read_space_from_file(self.kwargs['space'])


class DictionaryMixin:
    """"A mixin to read a dictionary.

    Dictionary is stored in Pandas Data frame inside of an .h5 file available at
    the key specified by the ``dictionary_key`` option.

    """
    global__dictionary = 'd', 'dictionary.h5', 'The input dictionary.'
    global__dictionary_key = '', 'dictionary', 'An identifier for the group in the store.'

    @Resource
    def dictionary(self):
        """ A dictionary.

        A dictionary is a DataFrame with the following columns:

        ngram
            the ngram (or the word)

        count
            the frequency of the ngrma

        It might also have these columns:

        tag
            the POS tag of ngram

        """
        return pd.read_hdf(self.kwargs['dictionary'], key=self.dictionary_key)
