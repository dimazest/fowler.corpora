Installation
============

.. warning:: This manual is in very early alpha!

    It has not been tested and many problems may appear during the deployment
    process.

You can deploy ``fowler.corpora`` in two ways: in your current operating
system, or inside of a virtual environment. While local deployment doesn't
compromise performance, it's much easier to deploy in an virtual environment.

Getting the deployment configuration:

.. code-block:: bash

    git clone https://github.com/dimazest/fc.git
    cd fc

Local install
-------------

You can use `saltstack`_ to install dependencies for you. But check it's
configuration is ``salt/`` to be sure that it doesn't wipe out your data:

.. _saltstack: http://docs.saltstack.com/en/latest/topics/installation/index.html

.. code-block:: bash

    # sudo is missing on purpose
    # just to make sure that you know what you are doing!
    salt-call --local state.highstate --config-dir salt

.. note:: Additional software

    In case you are using Ubuntu 13.10 (or Mint), you need to install some of the
    packages manually:

    .. code-block:: bash

        pip3 install --user numexp
        python3 ez_setup.py --user

Finally you are ready to deploy ``fowler.corpora``:

.. code-block:: bash

    python3 bootstrap.py
    bin/buildout


Virtual environment
-------------------

Alternatively to a local install, you can create a virtual machine. you need to
install `vagrant`_ and `virtualbox`_.

.. _vagrant: http://www.vagrantup.com/downloads.html
.. _virtualbox: https://www.virtualbox.org/wiki/Downloads

Then you are ready to spin up a virtual machine and ssh to it:

.. code-block:: bash

    vagrant up
    vagrant ssh

The final step
--------------

If everyrhing went fine, you should be able to run ``bin/corpora``:

.. code-block:: bash

    bin/corpora
    usage: corpora <command> [options]

    commands:

     bnc              Access to the BNC corpus.
     dictionary       Dictionary helpers.
     google-ngrams    The Google Books Ngram Viewer dataset helper routines.
     help             Show help for a given help topic or a help overview.
     ipcluster        Start IPYthon cluster.
     ipython          Start IPython.
     ms-paraphrase    Microsoft Research Paraphrase task described in [1, 2].
     notebook         Start IPython notebook.
     readline-folder  Concatinate files in the folder and print them.
     serafin03        Implementation of Latent Semantic Analysis for dialogue act classification.
     space            (no help text available)
     wordsim353       The WordSimilarity-353 Test.
     wsd              Implementation of common
