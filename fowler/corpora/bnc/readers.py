from nltk.corpus.reader.bnc import BNCCorpusReader

from .util import ccg_bnc_iter


class BNC:
    def __init__(self, paths, root, stem, tag_first_letter):
        self.paths = paths
        self.root = root
        self.stem = stem
        self.tag_first_letter = tag_first_letter

    def words(self, path):
        for word, tag in BNCCorpusReader(fileids=path, root=self.root).tagged_words(stem=self.stem):
            if self.tag_first_letter:
                tag = tag[0]

            yield word, tag


class BNC_CCG:
    def __init__(self, paths, stem, tag_first_letter):
        self.paths = paths
        self.stem = stem
        self.tag_first_letter = tag_first_letter

    def words(self, path):
        def word_tags(dependencies, tokens):
            for token in tokens.values():

                tag = token.tag[0] if self.tag_first_letter else token.tag

                if self.stem:
                    yield token.stem, tag
                else:
                    yield token.word, tag

        return ccg_bnc_iter(path, word_tags)
