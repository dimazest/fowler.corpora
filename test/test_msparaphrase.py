import pytest

from fowler.corpora.bnc.main import uri_to_corpus_reader


@pytest.fixture
def dataset(msparaphrase_path):
    return uri_to_corpus_reader(msparaphrase_path)


def read_file(dataset):
    df = dataset.read_file()

    train = df[df['split'] == 'train']
    assert len(train) == 4076

    test = df[df['split'] == 'test']
    assert len(test) == 1725


def test_dependency_graphs_pairs(dataset):
    g1, g2, score = next(dataset.dependency_graphs_pairs())

    assert score == 1


def test_words_by_document(dataset):
    words, = dataset.words_by_document(dataset.paths[0])

    assert next(words) == ('PCCW', 'PCCW', 'N')
    assert next(words) == ("'s", "'s", 'P')
    assert next(words) == ('chief', 'chief', 'N')
    assert next(words) == ('operating', 'operate', 'V')
    assert next(words) == ('officer', 'officer', 'N')

    list(words)
