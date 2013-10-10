from collections import Counter

from opster import command, dispatch

from .swda import CorpusReader


@command()
def hi():
    print("hi")


@command()
def transcripts(path):
    corpus = CorpusReader(path)

    for transcript in corpus.iter_transcripts(display_progress=False):
        for utterance in transcript.utterances:
            print('{u.caller}: {u.text}'.format(u=utterance))

        print()


def get_terms(corpus):
    for utterance in corpus.iter_utterances(display_progress=False):
        yield from utterance.pos_lemmas()  # noqa


@command()
def info(path, n=10, lemmas='True'):
    corpus = CorpusReader(path)

    terms = get_terms(corpus)
    if lemmas != 'True':
        terms = (term for term, tag in terms)

    counter = Counter(terms)

    for term, frequency in counter.most_common(int(n) or None):
        print(frequency, term)


@command()
def tags(path):
    corpus = CorpusReader(path)

    utterences = corpus.iter_utterances(display_progress=False)
    counter = Counter(u.act_tag for u in utterences)

    for tag, freq in counter.most_common():
        print(freq, tag)
