Quick start: Word similarity
============================

This tutorial will go trough the main steps involved in distributional
semantics experiments.

The task
--------

The [SimLex-999]_ data set consists of 353 word pairs judged by humans for
similarity. You can download the data set from `here`__.

::

    wget http://www.eecs.qmul.ac.uk/~dm303/static/data/SimLex-999/SimLex-999.txt

These are some of the records:

.. csv-table::
    :header-rows: 1

    word1,   word2,   POS,     SimLex999,       conc(w1),        conc(w2),        concQ,   Assoc(USF),      SimAssoc333,      SD(SimLex)
    old,     new,     A,       1.58,    2.72,    2.81,    2,       7.25,    1,       0.41
    smart,   intelligent,     A,       9.2,     1.75,    2.46,    1,       7.11,    1,       0.67
    hard,    difficult,       A,       8.77,    3.76,    2.21,    2,       5.94,    1,       1.19
    happy,   cheerful,        A,       9.55,    2.56,    2.34,    1,       5.85,    1,       2.18
    hard,    easy,    A,       0.95,    3.76,    2.07,    2,       5.82,    1,       0.93
    fast,    rapid,   A,       8.75,    3.32,    3.07,    2,       5.66,    1,       1.68
    happy,   glad,    A,       9.17,    2.56,    2.36,    1,       5.49,    1,       1.59
    short,   long,    A,       1.23,    3.61,    3.18,    2,       5.36,    1,       1.58

__ https://www.cl.cam.ac.uk/~fh295/SimLex-999.zip

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

..  Ignore

    To avoid the mess, the data is organized to the following folders:

    * ``corpora`` is the folder for different corpora distributions, for example
      ``corpora/BNC``.
    * ``downloads`` is for other resources, such as the wordsim 353 dataset.
    * ``data`` is the folder for the experiment data.

    If you use https://github.com/dimazest/fc deployment configuration, you
    should already have wordsim 353, otherwise you can get it from
    http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip

    It takes a while to process the BNC and needs a powerful machine. If you are
    curious and want to go trough the tutorial quickly on your laptop, tell corpora
    to process only a part of the BNC files by referring to the BNC corpus as::

        bnc://${PWD}/corpora/BNC/Texts/\?fileids=\\w/\\w[ADGR07]\\w*/\\w*\\.xml

    If you want to use the whole corpus, refer to the BNC as::

        bnc://${PWD}/corpora/BNC/Texts/

Use the ``-v`` flag to write logs to ``/tmp/fowler.log``. If you run
co-occurrence extraction on a laptop, to avoid lags, set the number of parallel
jobs less than the CPU cores, for example, for a 4 core machine ``-j 3``.

Extracting the data
-------------------

We will use the Brown to extract the co-occurrence matrix. The rows in the matrix
correspond to target words, while columns correspond to context words.

Targets
~~~~~~~

The target words are taken from the dataset:

.. code-block:: bash

    corpora bnc dictionary \
    --corpus simlex999://SimLex-999.txt?tagset=brown \
    --tag_first_letter \
    --stem -v -j 2 \
    -o dictionary_simlex_pos.h5

``dictionary_simlex_pos.h5`` is a `Pandas`_ `DataFrame`_ with the following columns:

    .. _Pandas: http://pandas.pydata.org/
    .. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

    ngram
        a word or a stem.

    tag
        its part of speech tag.

    count
        the frequency of the word.

We can access it the and extract the context words using IPython::

    corpora ipython

and executing the following code:

.. code-block:: python

    >>> import pandas as pd

    >>> dictionary = pd.read_hdf('dictionary_simlex_pos.h5', key='dictionary')
    >>> dictionary.head()
    ngram tag  count
    0   car   N     12
    1  door   N     10
    2  book   N      9
    3   arm   N      9
    4  give   V      9
    >>> dictionary[['ngram', 'tag']].to_csv('targets_simlex_pos.csv', index=False)

    >>> quit()


Contexts
~~~~~~~~

Context selection is more art than science, but a rather popular approach is to
select the 3000 most frequent words.

First we need to extract word frequencies:

.. code-block:: bash

    corpora bnc dictionary \
    --corpus brown:// \
    --tag_first_letter \
    --stem -v -j 2 \
    -o dictionary_brown_pos.h5

And again, a ``.csv`` file is created:

.. code-block:: python

    >>> import pandas as pd

    >>> dictionary = pd.read_hdf('dictionary_brown_pos.h5', key='dictionary')
    ngram tag  count
    0   the   A  69968
    1     ,   ,  58333
    2     .   .  49346
    3    of   I  36410
    4   and   C  28850
    >>> contexts = dictionary[:3000]
    >>> contexts[['ngram', 'tag']].to_csv('contexts_brown_pos_3000.csv', index=False)

The space
~~~~~~~~~

Now we are ready to extract the target-context co-occurrence frequencies and
get the first semantic space:

.. code-block:: bash

    corpora bnc cooccurrence \
    -t targets_simlex_pos.csv \
    -c contexts_brown_pos_3000.csv \
    --corpus brown:// \
    --stem -j 2 -v \
    --tag_first_letter \
    -o space_brown_simlex_3000.h5

Experiments
-----------

Now we are ready to run the first experiment:

.. code-block:: bash

    corpora wsd similarity \
    --space space_brown_simlex_3000.h5 \
    --dataset simlex999://SimLex-999.txt?tagset=brown \
    --composition_operator head \
    --output brown_simlex_3000.h5

    Spearman correlation (head), cosine): rho=-0.098, p=0.00203, support=999

And access the results:

.. code-block:: Python

    >>> import pandas as pd

    >>> results = pd.read_hdf('brown_simlex_3000.h5', key='dataset')
    >>> results.head(10)
        unit1           unit2       cos  inner_product  score
    0     (old )          (new )  0.000000              0   1.58
    1   (smart )  (intelligent )  0.000000              0   9.20
    2    (hard )    (difficult )  0.000000              0   8.77
    3   (happy )     (cheerful )  0.000000              0   9.55
    4    (hard )         (easy )  0.000000              0   0.95
    5    (fast )        (rapid )  0.825153           3314   8.75
    6   (happy )         (glad )  0.000000              0   9.17
    7   (short )         (long )  0.789176           8445   1.23
    8  (stupid )         (dumb )  0.000000              0   9.58
    9   (weird )      (strange )  0.000000              0   8.93

.. Ignore

    The score of -0.054 is very far fro the state-of-the-art, because of the tiny
    part of the corpus we've used.

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

    .. code-block:: python

        >>> import pandas as pd

        >>> pd.read_hdf('data/dictionary_bnc_pos.h5', key='dictionary').groupby('ngram').sum().sort('count', ascending=False).reset_index().to_hdf('data/dictionary_bnc.h5', 'dictionary', mode='w', complevel=9, complib='zlib')

        >>> quit()

    Now we are ready to weight the co-occurrence counts:

    .. code-block:: bash

        bin/corpora space pmi --column-dictionary data/dictionary_bnc_pos.h5 --dictionary data/dictionary_bnc.h5 \
        -s data/space_bnc_wordsim_3000.h5 -o data/space_bnc_wordsim_3000_ppmi.h5

    And run the experiment:

    .. code-block:: bash

        bin/corpora similarity wordsim353 -s data/space_bnc_wordsim_3000_ppmi.h5 \
        --alter_experiment_data

        Cosine similarity (Spearman): rho=0.032, p=0.55

    The small result is due to the small size of the corpus.

    Integration with IPython notebook
    ---------------------------------

    This IPython notebook :download:`quick_start_nb.ipynb <quick_start_nb.ipynb>`
    shows how ``corpora`` integrates with IPython. Copy the url to
    http://nbviewer.ipython.org to render it.

    Start IPython Notebook as:

    .. code-block:: bash

        bin/corpora notebook

    to have access to ``fowler.corpora``.

Conclusion
----------

A general workflow is the following:

1. Decide what the target words are.
2. Think of context words, possibly by extracting the (tagged) token counts from the corpus.
3. Extract the co-occurrence counts as an initial space.
4. Optionally modify the co-occurrence space, for example, by applying the PPMI weighting scheme.
5. Run an experiment.


References
----------

.. [SimLex-999]  Felix Hill, Roi Reichart and Anna Korhonen.
    `SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation`__.
    Computational Linguistics. 2015

    __ http://arxiv.org/abs/1408.3456v1

.. [Agirre09] Agirre, E., Alfonseca, E., Hall, K., Kravalova, J., Paşca, M., & Soroa,
    A. (2009, May). `A study on similarity and relatedness using distributional
    and WordNet-based approaches`__. In Proceedings of Human Language
    Technologies: The 2009 Annual Conference of the North American Chapter of
    the Association for Computational Linguistics (pp. 19-27). Association for
    Computational Linguistics.

    __ http://www.cs.brandeis.edu/~marc/misc/proceedings/naacl-hlt-2009/NAACLHLT09/pdf/NAACLHLT09003.pdf

.. [harris54] Z.S. Harris. 1954. Distributional structure. Word.
