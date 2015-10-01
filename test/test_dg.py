from nltk.parse.dependencygraph import DependencyGraph

from fowler.corpora.wsd.main import graph_signature

import pytest


@pytest.fixture
def dependency_graph(tree):
    return DependencyGraph(tree)


@pytest.mark.parametrize(
    ('tree', 'expected_result'),
    (
        (
            'green\tJ\t0\tROOT',
            (
                ('TOP', None), (
                    (('J', 'ROOT'), ()),
                )
            )
        ),
        (
            (
                '\t'.join(('John', 'N', '2', 'SBJ')),
                '\t'.join(('loves', 'V', '0', 'ROOT')),
                '\t'.join(('Mary', 'N', '2', 'OBJ')),
            ),
            (
                ('TOP', None),
                (
                    (
                        ('V', 'ROOT'),
                        (
                            (('N', 'SBJ'), ()),
                            (('N', 'OBJ'), ()),
                        ),
                    ),
                ),
            ),
        )
    ),
)
def test_graph_signature(dependency_graph, expected_result):
    result = graph_signature(dependency_graph)

    assert result == expected_result
