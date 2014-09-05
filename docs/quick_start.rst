Quick start: Word similarity
============================

This tutorial will go trough the main steps involved in distributional
semantics experiments.

The task
--------

The [wordsim353]_ data set consists of 353 word pairs judged by humans for
similarity. You can download the data set from `here`__. These are the first 9
records::

    curl -s http://www.eecs.qmul.ac.uk/~dm303/static/data/wordsim353/combined.csv | head
    Word 1,Word 2,Human (mean)
    love,sex,6.77
    tiger,cat,7.35
    tiger,tiger,10.00
    book,paper,7.46
    computer,keyboard,7.62
    computer,internet,7.58
    plane,car,5.77
    train,car,6.31
    telephone,communication,7.50

__ http://www.eecs.qmul.ac.uk/~dm303/static/data/wordsim353/combined.csv

Our task is to predict the human judgment given a pair of words from the
dataset. Refer, for example, to [Agirre09]_ for one way of solving it.

Idea
----

We are going to exploit Zellig Harris's intuition, that semantically similar
words tend to appear in similar contexts [harris54]_, in the following manner:
given a large piece of text, for every word we count its co-occurrence with
other words in a symmetric window of 5 (5 words before the word and 5 words
after). The word in the middle of a window is referred as the **target** word,
the words before and after as **context** words.

If we do this over the `British National Corpus`_ (BNC), set the target words
to:

.. _`British National Corpus`: http://www.natcorp.ox.ac.uk/

* ``Mary``
* ``John``
* ``girl``
* ``boy``
* ``idea``

and chose the following three words as context words:

* ``philosophy``
* ``book``
* ``school``

we produce this co-occurrence table

==== ========== ==== ======
\    philosophy book school
==== ========== ==== ======
Mary 0          10   22
John 4          60   59
girl 0          19   93
boy  0          12   146
idea 10         47   39
==== ========== ==== ======

``boy`` and ``girl`` get similar numbers, but different to ``idea``. If we
model word meaning as vectors in a highly dimensional vector space, where
dimensions are optionally labeled by the context words, we can measure the
similarity of a word pair as the distance between the corresponding vectors.

To see how good our similarity predictions are, we will use the Spearman
:math:`\rho` correlation.

Before we begin
---------------

To avoid the mess, the data is organized to the following folders:

* ``corpora`` is the folder for different corpora distributions, for example
  ``corpora/BNC``.
* ``downloads`` is for other resources, such as the wordsim 353 dataset.
* ``data`` is the folder for the experiment data.

If you use https://github.com/dimazest/fc deployment configuration, you
should already have wordsim 353, otherwise you can get it from
http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip


..    It takes a while to process the BNC and needs a powerful machine. If you
    are curious and want to go trough the tutorial quickly on your laptop, tell
    corpora to process only part of the BNC files by adding the following
    option::

..        --fileids='A/\w*/\w*\.xml'

Use the ``-v`` flag to write logs to ``/tmp/fowler.log``. If you run
co-occurrence extraction on a laptop, to avoid lags, set the number of parallel
jobs less than the CPU cores, for example, for a 4 core machine ``-j 3``.

Extracting the data
-------------------

We will use the BNC to extract the co-occurrence matrix. The rows in the matrix
correspond to target words, while columns correspond to context words.

Targets
~~~~~~~

The words in the wordsim 353 dataset are the target words. Here is a way to get
them:

.. code-block:: bash

    # Get the first colum
    cut downloads/wordsim353/combined.csv -d, -f 1 > t
    # Append the second column
    cut downloads/wordsim353/combined.csv -d, -f 2 >> t
    # The header
    echo ngram > data/targets_wordsim353.csv
    # Get rid of duplicates and the column names ("Word 1", "Word 2")
    # Lowercase the words and replace "troops" with its stem "troop"
    # This transformation is needed because word sems will be used to extract co-occurences.
    cat t | sort | uniq | grep -v Word | tr '[:upper:]' '[:lower:]' | sed -e 's/troops/troop/g' >> data/targets_wordsim353.csv
    rm t

Contexts
~~~~~~~~

Context selection is more art than science, but a rather popular approach is to
select the 2000 most frequent nouns, verbs, adjectives and adverbs, excluding
the 100 most frequent.

First we need to extract word frequencies:

.. code-block:: bash

    bin/corpora bnc dictionary \
    --corpus bnc://${PWD}/corpora/BNC/Texts/\?fileids=A/\\w*/\\w*\\.xml \
    -o data/dictionary_bnc_pos.h5 \
    --stem -v -j 3

``data/dictionary_bnc_pos.h5`` is a `Pandas`_ `DataFrame`_ with the following columns:

.. _Pandas: http://pandas.pydata.org/
.. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

ngram
    a word or a stem.

tag
    its part of speech tag. In the BNC, nous are tagged as ``SUBST``, verbs
    as ``VERB``, adjectives as ``ADV`` and adverbs as ``ADV``.

count
    the frequency of the word.

We can access it the and extract the context words using IPython::

    bin/corpora ipython

and executing the following code:

.. code-block:: python

    >>> import pandas as pd

    >>> dictionary = pd.read_hdf('data/dictionary_bnc_pos.h5', key='dictionary')
    >>> dictionary
           ngram   tag    count
    306889   the   ART  6042959
    45280      ,   PUN  5017057
    95027      .   PUN  4715135
    522342    be  VERB  4121594
    540719    of  PREP  3041681

    [5 rows x 3 columns]

    >>> #  We are interested only in 2000 most frequent (excluding the first 100)
    >>> #  nouns, verbs, adjectives and adverbs!
    >>> tags = dictionary['tag']
    >>> contexts = dictionary[(tags == 'SUBST') | (tags == 'VERB') | (tags == 'ADJ') | (tags == 'ADV')][101:2101]

    >>> contexts[['ngram', 'tag']].to_csv('data/contexts_bnc_pos_101-2101.csv', index=False)

    >>> quit()

The space
~~~~~~~~~

Now we are ready to extract the target-context co-occurrence frequencies and
get the first semantic space:

.. code-block:: bash

    bin/corpora bnc cooccurrence -t data/targets_wordsim353.csv -c data/contexts_bnc_pos_101-2101.csv \
    --bnc corpora/BNC/Texts/ -o data/space_bnc_wordsim_101-2101.h5 --stem

Experiments
-----------

Now we are ready to run the first experiment:

.. code-block:: bash

    bin/corpora wordsim353 evaluate -m data/space_bnc_wordsim_101-2101.h5
    ==================== ============== ===========
                Measure   Spearman rho     p-value
    ==================== ============== ===========
                 Cosine         0.350    1.357e-11
          Inner product        -0.035    5.098e-01
    ==================== ============== ===========

As you can see two similarity measures are used: one based on cosine distance
and other is Inner product. The score of 0.35 is not the state-of-the-art, but
for the raw co-occurrence counts it's pretty good.

Tuning
------

The artistic part of the experiment is to tweak the initial co-occurrence
counts. A common technique is to use positive pointwise mutual information (PPMI):

.. background and motivation

.. math::

    ppmi(t, c) = max(0, \log(\frac{p(t|c)}{p(c)p(t)})) = max(0, log(\frac{count(t, c)N}{count(t)count(c)}))

where :math:`count(t, c)` is the co-occurrence frequency of a target word with
a context word, :math:`count(t)` and :math:`count(c)` are the total number of
times the target word was seen in the corpus and the total number of times the
context word was seen in the corpus, :math:`N` is the total number of words.

So far we know the co-occurrence counts :math:`count(t, c)` from the space file
and the context counts :math:`count(c)` from the dictionary. Because our
contexts are part of speech tagged, while targets are not, we need to retrieve the counts for targets:

.. code-block:: bash

    bin/corpora bnc dictionary --bnc corpora/BNC/Texts/ -o data/dictionary_bnc.h5 --stem --omit-tags

Now we are ready to weight the co-occurrence counts:

.. code-block:: bash

    bin/corpora space pmi --column-dictionary data/dictionary_bnc_pos.h5 --dictionary data/dictionary_bnc.h5 \
    -m data/space_bnc_wordsim_101-2101.h5  -o data/space_bnc_wordsim_101-2101_ppmi.h5

And run the experiment:

.. code-block:: bash

    bin/corpora wordsim353 evaluate -m data/space_bnc_wordsim_101-2101_ppmi.h5
    ==================== ============== ===========
                Measure   Spearman rho     p-value
    ==================== ============== ===========
                 Cosine         0.024    6.585e-01
          Inner product        -0.048    3.708e-01
    ==================== ============== ===========

IPython notebook
----------------

This IPython notebook :download:`quick_start_nb.ipynb <quick_start_nb.ipynb>`
shows how ``corpora`` integrates with IPython. Copy the url to
http://nbviewer.ipython.org to render it.

References
----------

.. [wordsim353] Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud
    Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. `Placing search
    in context`__: the concept revisited. ACM Transactions on Information
    Systems, 20(1):116–131.

    __ http://www.cs.technion.ac.il/~gabr/papers/context_search.pdf

.. [Agirre09] Agirre, E., Alfonseca, E., Hall, K., Kravalova, J., Paşca, M., & Soroa,
    A. (2009, May). `A study on similarity and relatedness using distributional
    and WordNet-based approaches`__. In Proceedings of Human Language
    Technologies: The 2009 Annual Conference of the North American Chapter of
    the Association for Computational Linguistics (pp. 19-27). Association for
    Computational Linguistics.

    __ http://www.cs.brandeis.edu/~marc/misc/proceedings/naacl-hlt-2009/NAACLHLT09/pdf/NAACLHLT09003.pdf

.. [harris54] Z.S. Harris. 1954. Distributional structure. Word.
