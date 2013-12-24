from IPython.terminal.ipapp import launch_new_instance

import os
import sys
from itertools import islice

import fowler.corpora.dictionary.main as dictionary_main
import fowler.corpora.google_ngrams.main as google_ngrams_main
import fowler.corpora.serafin03.main as serafin03_main
import fowler.corpora.space.main as space_main
import fowler.corpora.wordsim353.main as wordsim353_main

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


@command()
def readline_folder(
    path,
    limit=('l', 0, 'Home many items to show, 0 shows all'),
):
    """Concatinate files in the folder and print them.

    Files might be compressed.

    """
    limit = limit or None

    with io_readline_folder(path) as lines:
        lines = islice(lines, limit)

        for line in lines:
            print(line.strip())


def ipython():
    os.environ['PYTHONPATH'] = ':'.join(sys.path)
    sys.exit(launch_new_instance())
