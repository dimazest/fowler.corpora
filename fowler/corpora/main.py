import warnings

with warnings.catch_warnings():
    import sklearn  # noqa


import os
import sys
from itertools import islice

from IPython.terminal.ipapp import launch_new_instance
from IPython.parallel.apps import ipclusterapp

import fowler.corpora.bnc.main as bnc_main
import fowler.corpora.dictionary.main as dictionary_main
import fowler.corpora.google_ngrams.main as google_ngrams_main
import fowler.corpora.ms_paraphrase.main as ms_paraphrase_main
import fowler.corpora.serafin03.main as serafin03_main
import fowler.corpora.space.main as space_main
import fowler.corpora.wordsim353.main as wordsim353_main
import fowler.corpora.wsd.main as wsd_main


from fowler.corpora.io import readline_folder as io_readline_folder

from .dispatcher import Dispatcher


dispatcher = Dispatcher()
command = dispatcher.command
dispatch = dispatcher.dispatch


dispatcher.nest(
    'serafin03',
    serafin03_main.dispatcher,
    serafin03_main.__doc__,
)

dispatcher.nest(
    'google-ngrams',
    google_ngrams_main.dispatcher,
    google_ngrams_main.__doc__,
)

dispatcher.nest(
    'wordsim353',
    wordsim353_main.dispatcher,
    wordsim353_main.__doc__,
)


dispatcher.nest(
    'dictionary',
    dictionary_main.dispatcher,
    dictionary_main.__doc__,
)


dispatcher.nest(
    'space',
    space_main.dispatcher,
    space_main.__doc__,
)

dispatcher.nest(
    'wsd',
    wsd_main.dispatcher,
    wsd_main.__doc__,
)

dispatcher.nest(
    'bnc',
    bnc_main.dispatcher,
    bnc_main.__doc__,
)

dispatcher.nest(
    'ms-paraphrase',
    ms_paraphrase_main.dispatcher,
    ms_paraphrase_main.__doc__,
)


@command()
def readline_folder(
    path=('p', '.', 'The folder to read files from.'),
    limit=('l', 0, 'Home many lines to show, 0 shows all.'),
):
    """Concatinate files in the folder and print them.

    Files might be compressed.

    """
    limit = limit or None

    with io_readline_folder(path) as lines:
        lines = islice(lines, limit)

        for line in lines:
            print(line.strip())


@command()
def ipython():
    """Start IPython."""
    os.environ['PYTHONPATH'] = ':'.join(sys.path)
    sys.exit(launch_new_instance(argv=[]))


@command()
def notebook():
    """Start IPython notebook."""
    os.environ['PYTHONPATH'] = ':'.join(sys.path)
    sys.exit(launch_new_instance(argv=['notebook']))


@command()
def ipcluster():
    """Start IPYthon cluster."""
    import logging
    logger = logging.getLogger()
    logger.info('Hellow from ipcluster')

    os.environ['PYTHONPATH'] = ':'.join(sys.path)
    ipclusterapp.launch_new_instance(argv='start -n 10'.split())
