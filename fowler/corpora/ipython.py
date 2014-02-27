import warnings
from imp import reload

from IPython.core.magic import Magics, magics_class, line_magic


@magics_class
class CorporaMagics(Magics):

    @line_magic
    def corpora(self, parameter_s=''):
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
