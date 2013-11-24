import fowler.corpora.serafim03.main as serafim03_main
import fowler.corpora.google_ngrams.main as google_ngrams_main

from .options import Dispatcher


dispatcher = Dispatcher()
command = dispatcher.command
dispatch = dispatcher.dispatch


dispatcher.nest(
    'serafin03',
    serafim03_main.dispatcher,
    serafim03_main.__doc__,
)

dispatcher.nest(
    'google-ngrams',
    google_ngrams_main.dispatcher,
    serafim03_main.__doc__,
)
