"""Helpers to get The Google Books Ngram Viewer dataset.

.. warning:: The script is in very early stage and is targeted to downlad the English
Version 20120701 dataset.

"""
import json
from collections import Counter
from itertools import product


from opster import Dispatcher
from py.path import local

from google_ngram_downloader import readline_google_store


dispatcher = Dispatcher()
command = dispatcher.command


@command()
def cooccurrence(
    ngram_len=('n', 2, 'The length of ngrams to be downloaded.'),
    output=('o', 'downloads/google_ngrams/{ngram_len}_cooccurrence_matrix/', 'The destination folder for downloaded files.'),
    verbose=('v', False, 'Be verbose.'),
):
    assert ngram_len > 1
    middle_index = ngram_len // 2
    output_dir = local(output.format(ngram_len=ngram_len))

    for fname, _, records in readline_google_store(ngram_len, verbose=verbose):
        output_file = output_dir.join(fname + '.json')

        if output_file.check():
            continue

        cooc = Counter()

        for i, record in enumerate(records, start=1):

            ngram = record.ngram.split()
            # Filter out any annotations. E.g. removes `_NUM` from  `+32_NUM`
            ngram = tuple(n.split('_')[0] for n in ngram)
            count = int(record.match_count)

            item = ngram[middle_index]
            context = ngram[:middle_index] + ngram[middle_index + 1:]

            cooc.update({p: count for p in product([item], context)})

        with output_file.open('w') as f:
            json.dump(cooc, f)



