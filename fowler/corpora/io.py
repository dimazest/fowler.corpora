"""IO functions."""
import fileinput
import contextlib
import codecs

from py.path import local


@contextlib.contextmanager
def readline_folder(input_dir=None):
    """Open all files in the `input_dir` as one big file.

    Uncompresses the files on the fly if needed.

    """
    if input_dir:
        file_names = [str(n) for n in local(input_dir).visit() if n.check(file=True, exists=True)]
    else:
        file_names = []

    with contextlib.closing(
        fileinput.FileInput(
            file_names,
            openhook=fileinput.hook_compressed,
            mode='rb',
        ),
    ) as lines:
        yield codecs.iterdecode(lines, 'utf-8')
