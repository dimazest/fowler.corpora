Installation
============

It's recommended to use Anaconda and install some packages with it. Refer to
`miniconda homepage`__ for links to installers for other platforms.

__ http://conda.pydata.org/miniconda.html

.. code-block:: bash

    # Install miniconda
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    sh Miniconda3-latest-MacOSX-x86_64.sh -b

    # Conda-install some packages
    wget https://bitbucket.org/dimazest/phd-buildout/raw/tip/requirements.txt
    ~/miniconda3/bin/conda install -c https://conda.anaconda.org/dimazest --file requirements.txt pip

You also need NLK data:

.. code-block::

    ~/miniconda3/bin/python 'import nltk; nltk.download("brown")'

Tha package itself
------------------

The package is available on `PyPi
<https://pypi.python.org/pypi/fowler.corpora>`_ and can be isntalled with pip:

.. code-block:: bash

    ~/miniconda3/bin/pip install fowler.corpora

It's also possible to isnstal a development version right from `GitHub
<https://github.com/dimazest/fowler.corpora/>`_:

.. code-block:: bash

    ~/miniconda3/bin/pip install https://github.com/dimazest/fowler.corpora/archive/master.zip


The final step
--------------

Run the package to see whether it works.

.. code-block:: bash

    ~/miniconda3/bin/corpora -h
    usage: corpora <command> [options]

    commands:
     help             Show help for a given help topic or a help overview.
     ...
     ...
     ...
