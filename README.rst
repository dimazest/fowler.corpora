fowler.corpora
==============

``fowler.corpora`` is software to create vector space models for distributional
semantics.

It is possible to instantiate a vector space from

* Brown corpus
* British National Corpus
* ukWaC and WaCkypedia

The weighting schemes include:

* PMI
* PPMI
* nITTF

The implemented experiments are:

* Word similarity

  * `SimLex-999 <http://www.cl.cam.ac.uk/~fh295/simlex.html>`_
  * `Men <http://clic.cimec.unitn.it/~elia.bruni/MEN>`_

* Sentence similarity

  * `KS14 <http://compling.eecs.qmul.ac.uk/wp-content/uploads/2015/07/KS2014.txt>`_

Chnagelog
---------

0.3
~~~

* Documentation update: installation instructions, similarity experiment quick
  start.
* Correlation and Eucliedean similarities are computed.
* PMI variants and parameters.
* Frobenious operators.
* Word2vec space import.

