Quick start: Measuring word similarity
======================================

This tutorial will go trough the main steps involved in distributional
semantics experiments.

The task
--------

The data set consists of 353 word pairs judged by humans for similarity
[wordsim353]_. You can download the data set from `here`__. These are the first
9 records::

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

Method
------

We are going to exploit Zellig Harris's intuition, that semantically similar
words tend to appear in similar contexts [harris54] in the following manner.
Given a large piece of text, for every word we count its co-occurrence with
other words in a symmetric window of 5 (5 words before the word and 5 words
after). The word in the middle of a window is referred as the **target** word,
the words before and after as **context** words.

If we do this over the `British National Corpus`_, set the target words to:

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
Word philosophy book school
==== ========== ==== ======
Mary 0          10   22
John 4          60   59
girl 0          19   93
boy  0          12   146
idea 10         47   39
==== ========== ==== ======

``boy`` and ``girl`` get similar numbers, but different to ``idea``, which
seems to fit the task. If we model word meaning as vectors in a highly
dimensional vector space, where dimensions are optionally labeled by the
context words, we can use the similarity of a word pair with the distance
between the corresponding vectors.

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
