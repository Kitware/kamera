#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import errno

import wx


def unclip_static_text(window):
    """Recompute StaticText best sizes so enlarged fonts are not clipped.

    wxFormBuilder enlarges title fonts *after* the StaticText controls are
    created, so under wxPython Phoenix/GTK the labels keep a best size computed
    with the original (smaller) font and the enlarged text gets clipped. Walk
    the window's child controls, recompute each label's best size with its
    final font, and re-layout. Should be invoked once the frame has been
    realized (e.g. via ``wx.CallAfter``).

    :param window: Top-level window (frame/panel) to fix up.
    :type window: wx.Window
    """
    def _walk(parent):
        for child in parent.GetChildren():
            if isinstance(child, wx.StaticText):
                child.InvalidateBestSize()
                child.SetMinSize(child.GetBestSize())
            _walk(child)
        # Re-layout each container (not just the top window) so labels are
        # repositioned (e.g. recentered) to the current container size.
        if parent.GetSizer() is not None:
            parent.Layout()

    _walk(window)


def refit_label(static_text):
    """Resize a StaticText to its current text and re-center it.

    A variable-length label keeps a min size pinned to an earlier value, which
    mis-centers the control once the text changes. Reset it to the current text
    and relayout the parent.
    """
    static_text.InvalidateBestSize()
    static_text.SetMinSize(static_text.GetBestSize())
    parent = static_text.GetParent()
    if parent is not None:
        parent.Layout()


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
