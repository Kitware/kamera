import os
import sys
import struct
import fcntl
import numpy as np



def write_blocking(data, i, basepath='/mnt/ram/rgb/img'):
    """ This blocks if any reader has LOCK_SH on the file. This isn't the best implementation, since
    a reader can hang without crashing, tying up the resource (crashing frees the fd).
    """
    name = basepath + '{}.bin'.format(i)
    with open(name, 'wb+') as fp:
        fcntl.lockf(fp.fileno(), fcntl.LOCK_EX)  # blocks until EX is released or fp is closed
        fp.write(data)


def read_nonblocking(name):
    """ This throws an IOError if a lock cannot be acquired. This shouldn't occur and is therefore exceptional"""
    with open(name, 'rb') as fp:
        fcntl.lockf(fp.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)  # throws error, this should never happen
        data = fp.read()



class JournalRingBuffer(object):
    """ work in progress"""
    def __init__(self, basepath='/mnt/ram/rgb/', N=8):
        self.basepath = basepath
        self.idx_path = os.path.join(basepath, 'index.bin')
        self.idx = 0

    def write(self):
        pass
    def open_for_reading(self, i):
        """ This throws an IOError if a lock cannot be acquired. This shouldn't occur and is therefore exceptional"""

        mask = 1 << i
        with open(self.idx_path, 'wb+') as fp:
            fcntl.lockf(fp.fileno(), fcntl.LOCK_EX)  # blocking - idx reads should be quick
            writes, reads = struct.unpack('HH', fp.read())
            if writes & mask:
                raise IOError('cannot read from file, it is currently being written')
            reads |= mask
            fp.seek(0)
            fp.write(struct.pack('HH', writes, reads))


        name = os.path.join(self.basepath, '{}'.format(i))
        with open(name, 'rb') as fp:
            fcntl.lockf(fp.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)  # throws error, this should never happen
            data = fp.read()


class MmapImageMsg(object):
    def __init__(self, filepath, shape, mode='r'):
        self.filepath = filepath
        self.shape = shape
        self.mm = None
        self.mode = mode

    def copy(self):
        return self.mm[:]

    def pre_hook(self):
        self.mm = np.memmap(self.filepath, dtype='uint8', mode=self.mode, shape=self.shape)

    def close(self):
        del self.mm

    def __enter__(self):
        self.pre_hook()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


