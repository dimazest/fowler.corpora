from docutils.examples import html_body
from progress.bar import Bar


def inside_ipython():
    try:
        return __IPYTHON__
    except NameError:
        pass


def display(value):
    if inside_ipython():
        from IPython.display import display as ipython_display, HTML
        if isinstance(value, str):
            ipython_display(HTML(rst_to_html(value)))
        else:
            ipython_display(value)
    else:
        print(value)


def rst_to_html(value):
    return html_body(value)


class Worker:
    def __init__(self, pool, show_progress_bar=False):
        self.show_progress_bar = show_progress_bar
        self.pool = pool

    def progressify(self, iterable, description, max=None):
        if self.show_progress_bar:
            return Bar(
                description,
                max=max,
                suffix='%(index)d/%(max)d, elapsed: %(elapsed_td)s',
            ).iter(
                iterable
            )
        else:
            return iterable
