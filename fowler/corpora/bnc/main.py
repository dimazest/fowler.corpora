"""Access to the BNC corpus.

You can obtain the full version of the BNC corpus at
http://www.ota.ox.ac.uk/desc/2554

"""
import logging
from collections import Counter, OrderedDict
from itertools import chain

from more_itertools import chunked

from nltk.corpus.reader.bnc import BNCCorpusReader

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceCreationMixin
from .util import count_cooccurrence


logger = logging.getLogger(__name__)


class BNCDispatcher(Dispatcher, SpaceCreationMixin):
    """BNC dispathcer."""

    global__bnc = '', 'corpora/BNC/Texts', 'Path to the BNC corpus.'
    global__fileids = '', r'[A-K]/\w*/\w*\.xml', 'Files to be read in the corpus.'

    @Resource
    def bnc(self):
        """BNC corpus reader."""
        root = self.kwargs['bnc']
        return BNCCorpusReader(root=root, fileids=self.fileids)


dispatcher = BNCDispatcher()
command = dispatcher.command


def bnc_cooccurrence(args):
    """Count word couccurrence in a BNC file."""
    root, fileids, window_size = args

    logger.debug('Processing %s', fileids)

    return count_cooccurrence(
        BNCCorpusReader(root=root, fileids=fileids).words(),
        window_size=window_size,
    )


class Index(OrderedDict):

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            id_ = len(self)
            self[key] = id_
            return id_


def sum_counters(args):
    if len(args) == 1:
        return args[0]

    logger.debug('Summing up %d counters.', len(args))
    return sum(args, Counter())


@command()
def cooccurrence(
    bnc,
    pool,
    window_size=('', 5, 'Window size.'),
    chunk_size=('', 7, 'Length of the chunk at the reduce stage.'),
):
    """Build the co-occurrence matrix."""
    index = Index()
    result = Counter()

    for fileids_chunk in chunked(bnc.fileids(), 100):

        imap_result = pool.imap_unordered(
            bnc_cooccurrence,
            ((bnc.root, fileids, window_size) for fileids in fileids_chunk)
        )

        # It would be nice to do it in parallel as well, but then `index` has to be shared.
        # Another possibility is to use redis.
        chunk_result = chain(
            (
                Counter(dict(((index[t], index[c]), n) for t, c, n in r))
            )
            for r in imap_result
        )

        while True:
            chunk_result = chunked(chunk_result, chunk_size)

            first = next(chunk_result)
            if len(first) == 1:
                logger.debug('Got results for a chunk.')
                result += first[0]
                break

            chunk_result = pool.imap_unordered(sum_counters, chain([first], chunk_result))

        logger.debug('There are %d cooccurrence records so far.', len(result))
        logger.debug('Index size is %d.', len(index))

    print(len(index))
