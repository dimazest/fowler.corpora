import inspect
import logging

import opster


class Dispatcher(opster.Dispatcher):
    def __init__(self, globaloptions=tuple(), middleware_hook=None):
        globaloptions = (
            tuple(globaloptions) +
            (
                ('v', 'verbose', False, 'Be verbose'),
            )
        )

        self.middleware_hook = middleware_hook

        super(Dispatcher, self).__init__(
            globaloptions=globaloptions,
            middleware=self._middleware,
        )

    def _middleware(self, func):
        def wrapper(*args, **kwargs):
            if func.__name__ == 'help_inner':
                return func(*args, **kwargs)

            f_args = inspect.getargspec(func)[0]

            verbose = kwargs.pop('verbose')

            logger = logging.getLogger('fowler')
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)-6s: %(name)s - %(levelname)s - %(message)s')

            handler.setFormatter(formatter)
            logger.addHandler(handler)

            if verbose:
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(logging.CRITICAL)

            if self.middleware_hook:
                self.middleware_hook(f_args, kwargs)

            return func(*args, **kwargs)

        return wrapper
