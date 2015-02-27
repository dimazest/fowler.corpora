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
        filename='/tmp/fowler.corpora_worker',
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

    def run(self, remote_func, iterable, init_func=None, verbose=True):

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

            if verbose and item[0] == 'message':
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

                if verbose:
                    logger.debug('Sent task %r to %s', task, channel.gateway.id)


def initialize_channel(channel):
    import pickle

    channel.send(('message', 'ready'))

    message, data = channel.receive()

    if (message, data) == ('message', 'terminate'):
        return message, data

    assert message == 'data'

    return message, pickle.loads(data)


def sum_folder(channel):
    import pickle
    from more_itertools import peekable

    from fowler.corpora.execnet import initialize_channel

    _, data = initialize_channel(channel)

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

            intermediate_results = peekable(folder(data, **kwargs))

            if intermediate_results:
                if result is None:
                    result = next(intermediate_results)

                for r in intermediate_results:
                    result = result.add(r, fill_value=0)

        channel.send(('message', 'send_next'))


def verb_space_builder(channel):
    import pickle
    from scipy import sparse

    from fowler.corpora.execnet import logger, initialize_channel
    from fowler.corpora.models import read_space_from_file

    _, data = initialize_channel(channel)
    space = read_space_from_file(data['space_file'])

    result = {}
    for item in channel:

        if item == ('message', 'terminate'):
            if result:
                channel.send(('result', pickle.dumps(result)))
            break

        type_, data = item
        if type_ == 'task':
            # for (subj_stem, subj_tag, obj_stem, obj_tag), group in pickle.loads(data):

            # (subj_stem, subj_tag, obj_stem, obj_tag), group = pickle.loads(data)
            (verb_stem, verb_tag), group = pickle.loads(data)

            logger.debug(
                'Processing verb %s_%s with %s argument pairs.',
                verb_stem,
                verb_tag,
                len(group),
                )

            for subj_stem, subj_tag, obj_stem, obj_tag, count in group[['subj_stem', 'subj_tag', 'obj_stem', 'obj_tag', 'count']].values:

                try:
                    subject_vector = space[subj_stem, subj_tag]
                    object_vector = space[obj_stem, obj_tag]
                except KeyError:
                    # logger.exception('Could not retrieve an argument vector.')
                    continue

                if not subject_vector.size:
                    logger.warning('Subject %s %s is empty!', subj_stem, subj_tag)
                    continue

                if not object_vector.size:
                    logger.warning('Object %s %s is empty!', obj_stem, obj_tag)
                    continue

                subject_object_tensor = sparse.kron(subject_vector, object_vector)
                t = subject_object_tensor * count

                if verb_stem not in result:
                    result[verb_stem, verb_tag] = t
                else:
                    result[verb_stem, verb_tag] += t

        channel.send(('message', 'send_next'))
