Fowler.Corpora
==============

.. image:: https://travis-ci.org/dimazest/fowler.corpora.png?branch=master
  :target: https://travis-ci.org/dimazest/fowler.corpora

.. image:: https://coveralls.io/repos/dimazest/fowler.corpora/badge.png?branch=master
  :target: https://coveralls.io/r/dimazest/fowler.corpora?branch=master


Development
-----------

To run the tests execute::

    python setup.py test

To run the tests on all supperoted Python versions and implementations run::

   tox

To install the package in an isolated environment for development run::

    virtualenv .
    source bin/activate

    pip install -e .
    pip install pytest>=2.4.2  # And any other tools you find useful.

Now you are ready to develop and test your changes::

    py.test test

If you want to execute some of the tests run (for example, IO related)::

   py.test test -k io

Read py.test documentation and have fun coding!
