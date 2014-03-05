"""An IPython extension that adds the %corpora magic.

To load the extension, run::

    %load_ext fowler.corpora.ipython

and then you can access the corpora command line entry point::

    %corpora -h

"""

import warnings
from imp import reload

from IPython.core.magic import Magics, magics_class, line_magic


@magics_class
class CorporaMagics(Magics):
    """The corpora IPython extension."""

    @line_magic
    def corpora(self, parameter_s=''):
        """The %corpora line magic.

        It provides an access to the corpora command line script. The main goal
        of this magic is to evaluate the command in a Python session, not in
        Bash. This gives ability to return objects from commands and reuse them
        afterwards inside of an IPython notebook. Also, the graphs are nicely
        rendered.

        In addition it reloads modules, so you can change the code and evaluate
        it inside IPython notebook without restarting it.

        """
        import sys

        for module in list(sys.modules.keys()):
            if module.startswith('fowler'):
                reload(sys.modules[module])

        from fowler.corpora import main

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return main.dispatch(parameter_s.split())


def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(CorporaMagics)
