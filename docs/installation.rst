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

Virtual environment
-------------------

First, you need to install `vagrant`_.

.. _vagrant: http://www.vagrantup.com/downloads.html

Then you are ready to spin up a virtual machine and ssh to it:

.. code-block:: bash

    # Start the virtual machine
    vagrant up

    # ssh to it
    vagrant ssh


Local install
-------------

You can use `saltstack`_ to install dependencies for you. But check it's
configuration is ``salt/`` to be sure that it doesn't wipe out your data:

.. _saltstack: http://docs.saltstack.com/en/latest/topics/installation/index.html

.. code-block:: bash

    # sudo is missing on purpose
    # just to make sure that you know what you are doing!
    salt-call --local state.highstate --config-dir salt

In case you are using Ubuntu 13.10 (or Mint), you need to install some of the
packages manually:

.. code-block:: bash

    pip3 install --user numexp
    python3 ez_setup.py --user

Finally you are ready to deploy the package:

.. code-block:: bash

    python3 bootstrap.py
    bin/buildout
