"""Microsoft Research Paraphrase task described in [1, 2].


This download [3] consists of data only: a text file containing 5800 pairs of
sentences which have been extracted from news sources on the web, along with
human annotations indicating whether each pair captures a paraphrase/semantic
equivalence relationship. No more than 1 sentence has been extracted from any
given news article. We have made a concerted effort to correctly associate with
each sentence information about its provenance and any associated information
about its author. If any attribution information is incorrect or missing, please
send email to billdol@microsoft.com and we will update the file.

[1] Dolan, B., Quirk, C., & Brockett, C. (2004, August). Unsupervised
construction of large paraphrase corpora: Exploiting massively parallel news
sources. In Proceedings of the 20th international conference on Computational
Linguistics (p. 350). Association for Computational Linguistics.

[2] DQuirk, C., Brockett, C., & Dolan, W. B. (2004, July). Monolingual Machine
Translation for Paraphrase Generation. In EMNLP (pp. 142-149).

[3] http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/

"""
import csv

import pandas as pd

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceMixin


class MSParaphraseDispatcher(Dispatcher, SpaceMixin):
    """ Microsoft Research Paraphrase Corpus task dispatcher."""

    global__ms_paraphrase_train = '', 'downloads/MSRParaphraseCorpus/msr_paraphrase_train.txt', 'Train split.'
    global__ms_paraphrase_test = '', 'downloads/MSRParaphraseCorpus/msr_paraphrase_train.txt', 'Test split.'

    def read_data(self, f_name):
        return pd.read_csv(f_name, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8-sig')

    @Resource
    def ms_paraphrase_train(self):
        self.read_data(self.kwargs['ms_paraphrase_train'])


dispatcher = MSParaphraseDispatcher()
command = dispatcher.command


@command()
def distributional(ms_paraphrase_train):
    print(ms_paraphrase_train)
