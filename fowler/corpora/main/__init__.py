from .options import Dispatcher
import fowler.corpora.serafim03.main as serafim03_main

dispatcher = Dispatcher()
command = dispatcher.command
dispatch = dispatcher.dispatch


dispatcher.nest(
    'serafin03',
    serafim03_main.dispatcher,
    serafim03_main.__doc__,
)


@command()
def info(cooccurrence_matrix):
    print(
        'The co-coocurance matrix shape is {m.shape}.'
        ''.format(m=cooccurrence_matrix)
        )
