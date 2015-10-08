import warnings

with warnings.catch_warnings():
    import sklearn  # noqa


import os
import sys
from itertools import islice

import fowler.corpora.bnc.main as bnc_main
import fowler.corpora.categorical.main as categorical_main
import fowler.corpora.dictionary.main as dictionary_main
import fowler.corpora.ms_paraphrase.main as ms_paraphrase_main
import fowler.corpora.serafin03.main as serafin03_main
import fowler.corpora.space.main as space_main
import fowler.corpora.word2vec.main as word2vec_main
import fowler.corpora.wsd.main as wsd_main
from fowler.corpora import produce

from fowler.corpora.io import readline_folder as io_readline_folder

from .dispatcher import Dispatcher


def dispatcher_factory():
    dispatcher = Dispatcher()
    command = dispatcher.command
    dispatch = dispatcher.dispatch

    dispatcher.nest(
        'serafin03',
        serafin03_main.dispatcher,
        serafin03_main.__doc__,
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

    dispatcher.nest(
        'categorical',
        categorical_main.dispatcher,
        categorical_main.__doc__,
    )

    dispatcher.nest(
        'word2vec',
        word2vec_main.dispatcher,
        word2vec_main.__doc__,
    )

    dispatcher.nest(
        'produce',
        produce.dispatcher,
        produce.__doc__,
    )

    return dispatcher

dispatcher = dispatcher_factory()
command = dispatcher.command
dispatch = dispatcher.dispatch


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
    from IPython.terminal.ipapp import launch_new_instance

    sys.exit(launch_new_instance(argv=[]))


@command()
def notebook():
    """Start Jupyter notebook."""
    import sys
    from notebook import notebookapp

    sys.argv[:] = ['fake']

    os.environ['PYTHONPATH'] = ':'.join(sys.path)
    sys.exit(notebookapp.launch_new_instance())
