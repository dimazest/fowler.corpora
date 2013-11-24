"""Helpers to get The Google Books Ngram Viewer dataset.

.. warning:: The script is in very early stage and is targeted to downlad the English
Version 20120701 dataset.

"""
import sys

import requests
from opster import Dispatcher
from py.path import local

from .util import get_indices

dispatcher = Dispatcher()
command = dispatcher.command

URL_TEMPLATE = 'http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-{}'
FILE_TEMPLATE = '{ngram_len}gram-{version}-{index}.gz'


@command()
def download(
    ngram_len=('n', 1, 'The length of ngrams to be downloaded.'),
    output=('o', 'downloads/google_ngrams/{ngram_len}', 'The destination folder for downoaded files.'),
    verbose=('v', False, 'Be verbose.'),
):
    """Download The Google Books Ngram Viewer dataset version 20120701."""

    version = '20120701'
    output = local(output.format(ngram_len=ngram_len))
    output.ensure_dir()

    session = requests.Session()

    for index in get_indices(ngram_len):
        fname = FILE_TEMPLATE.format(
            ngram_len=ngram_len,
            version=version,
            index=index,
        )

        output_path = output.join(fname)
        url = URL_TEMPLATE.format(fname)

        if verbose:
            sys.stderr.write(
                'Downloading {url} to {output_path} '
                ''.format(
                    url=url,
                    output_path=output_path,
                ),
            )

        with output_path.open('wb') as f:
            request = session.get(url, stream=True)

            for num, block in enumerate(request.iter_content(1024)):
                    if verbose and not divmod(num, 1024)[1]:
                        sys.stderr.write('.')
                        sys.stderr.flush()

                    if not block:
                        break

                    f.write(block)
            if verbose:
                sys.stderr.write('\n')

