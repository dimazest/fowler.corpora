fowler.corpora
==============

.. image:: https://travis-ci.org/dimazest/fowler.corpora.png?branch=master
  :target: https://travis-ci.org/dimazest/fowler.corpora

.. image:: https://coveralls.io/repos/dimazest/fowler.corpora/badge.png?branch=master
  :target: https://coveralls.io/r/dimazest/fowler.corpora?branch=master

``fowler.corpora`` is software to create vector space models for Distributional
semantics.

It is possible to instantiate a vector space from

* British National Corpus
* Google Books N-gram Corpus

The weighting schemes include:

* TF-IDF
* NMF
* PMI

The implemented experiments are:

* Word similarity (wordsim353)
* Dialog act tagging, using the Switchboard corpus http://www.eecs.qmul.ac.uk/~dm303/cvsc14.html
* Number of categorical composition experiments
