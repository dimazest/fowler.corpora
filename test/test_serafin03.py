from fowler.corpora.main import dispatcher

import pytest


@pytest.mark.xfail
def test_plain_lsa(swda_100_path, capsys):
    dispatcher.dispatch(
        'serafin03 plain-lsa -p {swda_100_path}'
        ''.format(
            swda_100_path=swda_100_path,
        ).split()
    )

    out, err = capsys.readouterr()

    assert out
    assert not err
