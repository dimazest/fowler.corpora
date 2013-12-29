from IPython.core.magic import Magics, magics_class, line_magic


@magics_class
class CorporaMagics(Magics):

    @line_magic
    def corpora(self, parameter_s=''):
        from fowler.corpora import main

        main.dispatch(parameter_s.split())


def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(CorporaMagics)
