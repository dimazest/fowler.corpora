from docutils.examples import html_body


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

