"""Helpers to get The Google Books Ngram Viewer dataset.

.. warning:: The script is in very early stage and is targeted to downlad the English
Version 20120701 dataset.

"""
import sys
import csv

from opster import Dispatcher
from py.path import local

from fowler.corpora.io import readline_folder
from .util import readline_google_store, iter_google_store, Record


dispatcher = Dispatcher()
command = dispatcher.command


@command()
def download(
    ngram_len=('n', 1, 'The length of ngrams to be downloaded.'),
    output=('o', 'downloads/google_ngrams/{ngram_len}', 'The destination folder for downoaded files.'),
    verbose=('v', False, 'Be verbose.'),
):
    """Download The Google Books Ngram Viewer dataset version 20120701."""
    output = local(output.format(ngram_len=ngram_len))
    output.ensure_dir()

    for fname, url, request in iter_google_store(ngram_len, verbose=verbose):
        with output.join(fname).open('wb') as f:

            for num, chunk in enumerate(request.iter_content(1024)):
                    if verbose and not divmod(num, 1024)[1]:
                        sys.stderr.write('.')
                        sys.stderr.flush()
                    f.write(chunk)


@command()
def cooccurrence(
    ngram_len=('n', 2, 'The length of ngrams to be downloaded.'),
    verbose=('v', False, 'Be verbose.'),
):
    assert ngram_len > 1

    for fname, url, lines in readline_google_store(ngram_len, verbose=verbose):
        for line in lines:
            print(line)

        break


@command()
def count_ngrams(path=None):
    """Count total number of ngrams in the collection."""
    with readline_folder(path) as data:

        reader = csv.reader(
            data,
            delimiter='\t',
        )

        records = map(Record._make, reader)
        ngrams = set(r.ngram for r in records)
        print(len(ngrams))
