# -*- coding: utf-8 -*-

from __future__ import division, print_function
import wx
from wxpython_gui.RemoteImagePanel import RemoteImagePanel
from wxpython_gui.UpdateImageThreadFit import UpdateImageThreadFit
from wxpython_gui.cfg import SYS_CFG


class RemoteImagePanelFit(RemoteImagePanel):
    """RemoteImagePanel with imagery fit to the size of the panel.

    """
    def __init__(self, wx_panel, srv_topic,
                 click_callback=None, status_static_text=None,
                 compressed=False, wx_histogram_panel=None, attrs=None):
        """
        :param wx_panel: Panel to add the image to.
        :type wx_panel: wx.Panel

        """
        super(RemoteImagePanelFit, self).__init__(wx_panel, srv_topic,
                                                  status_static_text,
                                                  compressed,
                                                  wx_histogram_panel,
                                                  attrs=attrs)
        self.click_callback = click_callback

    def start_image_thread(self):
        update_image_thread = UpdateImageThreadFit(self, self.srv_topic,
                                                   SYS_CFG["max_mpix"],
                                                   compressed=self.compressed)
        update_image_thread.start()
        self.update_image_thread = update_image_thread

    def process_clicked_point(self, pos, button):
        """
        :param pos: Raw image coordinates that were clicked.
        :type pos: 2-array

        :param button: The mouse button that was clicked (0 for left, 1 for
            right)
        :type button: 0 | 1

        """
        if self.click_callback is not None:
            self.click_callback(pos, button)

