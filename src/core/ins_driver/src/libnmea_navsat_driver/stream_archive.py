#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import errno
import struct
import warnings


def make_path(path, from_file=False, verbose=False):
    """
    Make a path, ignoring already-exists error. Python 2/3 compliant.
    Catch any errors generated, and skip it if it's EEXIST.
    :param path: Path to create
    :type path: str, pathlib.Path
    :param from_file: if true, treat path as a file path and create the basedir
    :return:
    """
    path = str(path)  # coerce pathlib.Path
    if path == '':
        raise ValueError("Path is empty string, cannot make dir.")

    if from_file:
        path = os.path.dirname(path)
    try:
        os.makedirs(path)
        if verbose:
            print('Created path: {}'.format(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        if verbose:
            print('Tried to create path, but exists: {}'.format(path))

def bundle_packet(buf):
    head = struct.pack('>BH', 1, len(buf))
    check = sum(bytearray(buf)) & 0xFFFF
    tail = struct.pack('>BHBB', 23, check, 13, 10)
    return head + buf + tail


def dumpbuf(filename, buf):
    make_path(filename, from_file=True)
    with open(filename, 'ab') as fp:
        fp.write(bundle_packet(buf))


def enumerate_packets(buf):
    """
    Unpack a binary stream packaged using dumpbuf

    Usage:
        for i, packet in enumerate_packets(buf):
            do_stuff(packet)


    Args:
        buf: Raw binary

    Returns:
        Generator which yields (i, packet)
    """
    count = 0
    while buf:
        sth, length = struct.unpack('>BH', buf[:3])
        if len(buf) < length:
            raise StopIteration('Incomplete packet')
        payload, tail = buf[3:length+3], buf[length+3:length+8]
        etb, checksum, cr, nl = struct.unpack('>BHBB', tail)

        check = sum(bytearray(payload)) & 0xFFFF
        if check != checksum:
            warnings.warn('Warning: invalid checksum', RuntimeWarning)
        buf = buf[length+8:]
        yield count, payload
        count += 1