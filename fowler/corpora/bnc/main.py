"""Access to the BNC corpus.

You can obtain the full version of the BNC corpus at
http://www.ota.ox.ac.uk/desc/2554

"""
import logging

from nltk.corpus.reader.bnc import BNCCorpusReader

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceCreationMixin


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


@command()
def cooccurrence(bnc):
    """Build the co-occurrence matrix."""
    words = bnc.tagged_words()

    print(len(words))
