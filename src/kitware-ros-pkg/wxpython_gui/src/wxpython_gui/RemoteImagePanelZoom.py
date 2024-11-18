# -*- coding: utf-8 -*-

from __future__ import division, print_function
import wx
import numpy as np
from wxpython_gui.RemoteImagePanel import RemoteImagePanel
from wxpython_gui.UpdateImageThreadZoom import UpdateImageThreadZoom
from wxpython_gui.cfg import SYS_CFG


class RemoteImagePanelZoom(RemoteImagePanel):
    """RemoteImagePanel with imagery zoomed to particular region.

    """
    def __init__(self, wx_panel, srv_topic, zoom_slider,
                 status_static_text=None, compressed=False,
                 wx_histogram_panel=None):
        """
        :param wx_panel: Panel to add the image to.
        :type wx_panel: wx.Panel

        """
        self._zoom = 100
        self._center = None
        super(RemoteImagePanelZoom, self).__init__(wx_panel,
                                                   srv_topic,
                                                   status_static_text,
                                                   compressed,
                                                   wx_histogram_panel)

        self.zoom_slider = zoom_slider
        self.wx_panel.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom_mouse_wheel)
        self.zoom_slider.Bind(wx.EVT_SCROLL, self.on_zoom_slider)

        # Use the default setting on the zoom slider to set initial zoom.
        self.on_zoom_slider()

    def start_image_thread(self):
        update_image_thread = UpdateImageThreadZoom(self, self.srv_topic,
                                                    self.get_center,
                                                    self.get_zoom,
                                                    SYS_CFG["max_mpix"],
                                                    compressed=False)
        update_image_thread.start()
        self.update_image_thread = update_image_thread

    def get_zoom(self):
        """Zoom percentage.

        """
        return self._zoom

    def set_zoom(self, zoom):
        """
        :param zoom: Zoom percentage.
        :type zoom: float
        """
        self._zoom = zoom
        self.update_all()

    def get_center(self):
        """Zoom percentage.

        """
        return self._center

    def set_center(self, center):
        self._center = center
        self.update_all()

    def process_clicked_point(self, pos, button):
        """
        :param pos: Raw image coordinates that were clicked.
        :type pos: 2-array

        :param button: The mouse button that was clicked (0 for left, 1 for
            right)
        :type button: 0 | 1

        """
        self.set_center(pos)

    def on_zoom_mouse_wheel(self, event=None):
        val = event.GetWheelRotation()
        if event.ShiftDown():
            change = 1.1
        else:
            change = 1.01

        if val > 0:
            self.on_zoom_up(change=change)
        if val < 0:
            self.on_zoom_down(change=change)

    def on_zoom_up(self, event=None, change=1.02):
        zoom = np.minimum(self._zoom*change, 2000)
        v = self.zoom_slider.SetValue(zoom)
        self.handle_updated_zoom(zoom)

    def on_zoom_down(self, event=None, change=1.02):
        zoom = np.maximum(self._zoom/change, 10)
        v = self.zoom_slider.SetValue(zoom)
        self.handle_updated_zoom(zoom)

    def handle_updated_zoom(self, zoom):
        self.set_zoom(zoom)

    def on_zoom_slider(self, event=None):
        """Slider takes values from 0 to 1000.
        """
        v = self.zoom_slider.GetValue()
        v = v/1000.0
        self.set_zoom(2*(1-v) + (1000-2)*v)

        if event is not None:
            self.update_all()