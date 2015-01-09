import logging
import pickle

import execnet

from more_itertools import peekable


logger = logging.getLogger(__name__)


def setup_logging(channel):
    import logging as this_logging
    from logging.handlers import RotatingFileHandler

    this_logging.captureWarnings(True)
    logger = this_logging.getLogger()
    handler = RotatingFileHandler(
        filename='/tmp/fowler.coropora_worker',
        backupCount=10,
    )
    formatter = this_logging.Formatter('%(asctime)-6s: %(name)s - %(levelname)s - %(process)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(this_logging.DEBUG)
    logger.debug('Logger is set up.')

    channel.send(('message', 'Logger is set up.'))


class ExecnetHub:
    def __init__(self, gateways):
        self.gateways = gateways

    def run(self, remote_func, iterable, init_func=None):

        channels = []
        for gw in self.gateways:
            l = gw.remote_exec(setup_logging)
            reply = l.receive()
            assert reply == ('message', 'Logger is set up.')

            ch = gw.remote_exec(remote_func)
            channels.append(ch)

        mch = execnet.MultiChannel(channels)

        endmarker = 'message', 'endmarker'

        q = mch.make_receive_queue(endmarker=endmarker)
        tasks = peekable(iterable)

        initialized = []
        terminated = []
        while True:
            channel, item = q.get()

            if item[0] == 'message':
                logger.debug(
                    'Gateway %s sent reply: %r',
                    channel.gateway.id,
                    item,
                )

            assert item[0] != 'exception'

            if item == endmarker:
                terminated.append(channel)
                logger.debug(
                    'Gateway %s:%s terminated. %s out of %s terminated.',
                    channel.gateway.id,
                    channel.gateway.spec,
                    len(terminated),
                    len(mch),
                )
                if len(terminated) == len(mch):
                    logger.debug('All geteways are terminated.')
                    break
                if tasks != endmarker:
                    raise RuntimeError('Someone exited before a termination request!')
                continue

            if item == ('message', 'ready'):
                logger.info(
                    'Gateway %s is ready',
                    channel.gateway.id,
                )

                if init_func:
                    init_func(channel)

                initialized.append(channel)

            if item[0] == 'result':
                type_, result = item
                result = pickle.loads(result)
                yield result

            if not tasks and len(initialized) == len(mch):
                termination_request = 'message', 'terminate'
                logger.debug(
                    'No tasks remain, '
                    'sending termination request %r to all.',
                    termination_request,
                )

                mch.send_each(termination_request)
                tasks = endmarker

            if tasks and tasks != endmarker:
                task = next(tasks)
                channel.send(('task', task))
                logger.debug('Sent task %r to %s', task, channel.gateway.id)


def sum_folder(channel):
    import pickle
    import logging as this_logging
    from more_itertools import peekable

    logger = this_logging.getLogger(__name__)

    channel.send(('message', 'ready'))

    message, data = channel.receive()

    if (message, data) == ('message', 'terminate'):
        return

    assert message == 'data'

    data = pickle.loads(data)

    kwargs = data.get('kwargs', {})
    instance = data['instance']
    folder_name = data['folder_name']
    folder = getattr(instance, folder_name)

    result = None
    for item in channel:

        if item == ('message', 'terminate'):
            if result is not None:
                channel.send(('result', pickle.dumps(result.reset_index())))
            break

        type_, data = item
        if type_ == 'task':

            try:
                intermediate_results = peekable(folder(data, **kwargs))
            except BaseException as e:
                logger.exception('Exception during execution, %s', e)
                raise

            if intermediate_results:
                if result is None:
                    result = next(intermediate_results)

                for r in intermediate_results:
                    result = result.add(r, fill_value=0)

            channel.send(('message', 'send_next'))
