import pytest

from fowler.corpora.bnc.main import uri_to_corpus_reader


@pytest.fixture
def dataset(ANDailment_path):
    return uri_to_corpus_reader(ANDailment_path)


def test_read_file(dataset):
    df = dataset.read_file()

    assert len(df) == 462


def test_dependency_graphs_pairs(dataset):
    g1, g2, score = next(dataset.dependency_graphs_pairs())

    assert score == 0


def test_words_by_document(dataset):
    words, = dataset.words_by_document(dataset.paths[0])

    assert next(words) == ('entrepreneur', 'entrepreneur', 'N')

    assert list(words)
