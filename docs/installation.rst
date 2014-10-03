Installation
============

.. warning:: This manual is in a very early alpha state!

    It has not been tested and many problems may appear during the deployment
    process.

You can deploy ``fowler.corpora`` in two ways: in your current operating
system, or inside of a virtual environment. While local deployment doesn't
compromise performance, it's much easier to deploy in an virtual environment.

Getting the deployment configuration:

.. code-block:: bash

    git clone https://github.com/dimazest/fc.git
    cd fc

System packages
---------------

Some systme libraries are needed to run the software. There are many ways to do
it, here are some:

Mac OS X with Macports
~~~~~~~~~~~~~~~~~~~~~~

Macports is a preferd package manager on a Mac to deploy ``fowler.corpora``
mainly because it provides packages for ``pandas``. ``atlas`` and friends.

.. code-block:: bash

    sudo port install py33-scikit-learn py33-pandas py33-matplotlib


Virtual environment
~~~~~~~~~~~~~~~~~~~

Alternatively to a local install, you can create a virtual machine. you need to
install `vagrant`_ and `virtualbox`_.

.. _vagrant: http://www.vagrantup.com/downloads.html
.. _virtualbox: https://www.virtualbox.org/wiki/Downloads

Then you are ready to spin up a virtual machine and ssh to it:

.. code-block:: bash

    vagrant up
    vagrant ssh

    cd /vagrant  # The directry is shared between the virtual machine and the host OS.

The final step
--------------

.. code-block:: bash

    python3.3 bootstrap.py
    bin/buildout

If everyrhing went fine, you should be able to run ``bin/corpora``:

.. code-block:: bash

    bin/corpora
    usage: corpora <command> [options]

    commands:

     help             Show help for a given help topic or a help overview.
     ...
     ...
     ...
