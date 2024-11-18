# -*- coding: utf-8 -*-

from __future__ import division, print_function
import wx
import threading
import numpy as np
import cv2
import time
from wxpython_gui.cfg import SYS_CFG, format_status, BRIGHT_RED

class RemoteImagePanel(object):
    """Provide image updates to a wx.Panel using a remote imagery request.

    Attributes:
    homography - the homography applied to the image returned from
        UpdateImageThread so that it fits inside the current panel. Generally,
        the resolution of the image requested from the remote server will be
        chosen to match the panel resolution so that resizing is not required.
        However, if the panel is resized before a new request can be made and
        the updated image received, the image is resized using this homography
        for so that it still fits in the panels. Also, if the panel size
        exceeds the resolution limits set by 'max_mpix', this homography will
        be used to enlarge the image received from the remote server to fit the
        panel.
    remote_homography -
    """
    def __init__(self, wx_panel, srv_topic,
                 status_static_text=None, compressed=False,
                 wx_histogram_panel=None, attrs=None):
        """
        :param wx_panel: Panel to add the image to.
        :type wx_panel: wx.Panel

        """
        self.wx_panel = wx_panel
        self.raw_image = None
        self.image = None
        self.raw_image_height = None
        self.raw_image_width = None
        self.panel_image_height = None
        self.panel_image_width = None
        self.homography = None
        self.remote_homography = None
        self.inverse_homography = None
        self.inverse_remote_homography = None
        self.wx_bitmap = None
        self.wx_histogram_panel = wx_histogram_panel
        self._histogram = None
        self._stop = False
        self.compressed = compressed
        self.last_update = None
        self.needs_update = True
        self.update_image_thread = None # type: UpdateImageThread
        self.attrs = attrs or {}
        #TODO: move image processing to service

        # Lock on access to self.raw_image.
        self.raw_image_lock = threading.RLock()

        self.wx_panel.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.status_static_text = status_static_text
        self.update_status_msg(format_status())
        self.srv_topic = srv_topic

        #self.start_image_thread()

        # ------------------------ Bind Events -------------------------------
        self.wx_panel.Bind(wx.EVT_LEFT_DOWN, self.on_click)
        self.wx_panel.Bind(wx.EVT_RIGHT_DOWN, self.on_click)
        self.wx_panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.wx_panel.Bind(wx.EVT_SIZE, self.on_size)
        # --------------------------------------------------------------------

    def start_image_thread(self):
        raise NotImplementedError('Subclass must define this method')

    def invalidate_cache(self):
        """ See UpdateImageThread.invalidate_cache"""
        self.update_image_thread.invalidate_cache()

    def update_remote_homography(self, remote_homography):
        """Called by thread after providing an updated image.

        """
        self.remote_homography = remote_homography
        self.inverse_remote_homography = np.linalg.inv(remote_homography)

    def update_raw_image(self, raw_image):
        """Replace raw_image and update the rendered view in the panel.

        """
        if self._stop: return None  # Check for a request to stop.
        with self.raw_image_lock:
            self.raw_image = raw_image
            self.update_all()

            if self.wx_histogram_panel is not None:
                self.generate_histogram()

            self.last_update = time.time()

    def on_size(self, event):
        """Called on event wx.EVT_SIZE.

        """
        self.update_all()

    def update_all_if_needed(self):
        if self._stop or self.needs_update is False: return None  # Check for a request to stop.
        # self.invalidate_cache()
        self.needs_update = False
        with self.raw_image_lock:
            if self.raw_image is not None:
                #print('on_size')
                panel_width, panel_height = self.wx_panel.GetSize()
                self.wx_image = wx.EmptyImage(panel_width, panel_height)
                self.update_homography()
                self.update_inverse_homography()
                self.warp_image()
                self.wx_panel.Refresh(True)
            else:
                self.wx_bitmap = None
                self._histogram = None
                self.wx_panel.Refresh(True)

    def update_all(self):
        self.needs_update = True

    def update_homography(self):
        """Update homography mapping self.raw_image to the panel.

        """
        panel_width, panel_height = self.wx_panel.GetSize()
        im_height, im_width = self.raw_image.shape[:2]

        """
        if im_width/im_height > panel_width/panel_height:
            # Side edges of image should hit the edges of the panel.
            s = panel_width/im_width
            y = (panel_height-s*im_height)/2
            self.homography = np.array([[s,0,0],[0,s,y],[0,0,1]])
        else:
            # Top edges of image should hit the edges of the panel.
            s = panel_height/im_height
            x = (panel_width-s*im_width)/2
            self.homography = np.array([[s,0,x],[0,s,0],[0,0,1]])
        """
        s = min(panel_height/im_height, panel_width/im_width)
        self.homography = np.array([[s,0,0],[0,s,0],[0,0,1]])

        corner_pts = np.array([[0,0,1],[0,im_height,1], [im_width,0,1],
                               [im_width,im_height,1]]).T
        corner_pts = np.dot(self.homography, corner_pts)
        corner_pts = corner_pts[:2]/corner_pts[2]
        self.panel_image_width = int(np.floor(max(corner_pts[0])))
        self.panel_image_height = int(np.floor(max(corner_pts[1])))

    def update_inverse_homography(self):
        """
        Calculate inverse of the homography.

        """
        if self.homography is None:
            return None
        else:
            self.inverse_homography = np.linalg.inv(self.homography)

    def warp_image(self):
        """Apply homography.

        """
        if self.raw_image is not None and self.inverse_homography is not None:
            panel_width, panel_height = self.wx_panel.GetSize()

            # Set linear interpolation.
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP

            image = cv2.warpPerspective(self.raw_image,
                                        self.inverse_homography,
                                        dsize=(self.panel_image_width,
                                               self.panel_image_height),
                                               flags=flags)

            wx_image = wx.EmptyImage(self.panel_image_width,
                                     self.panel_image_height)
            try:
                wx_image.SetData(image.tostring())
            except ValueError as err:
                raise ValueError('Shape: {}  Chan {} \n {}'.format(image.shape, self.raw_image.dtype, err))
            self.wx_bitmap = wx_image.ConvertToBitmap()
        else:
            self.wx_bitmap = None

    def on_click(self, event):
        """Called on events wx.EVT_RIGHT_DOWN or wx.EVT_LEFT_DOWN.

        """
        # self.invalidate_cache()
        if self.raw_image is not None and \
           self.inverse_remote_homography is not None:
            pos = list(event.GetPosition())
            panel_width, panel_height = self.wx_panel.GetSize()
            pos[0] -= (panel_width - self.panel_image_width)//2
            pos[1] -= (panel_height - self.panel_image_height)//2

            pos = np.dot(self.inverse_homography, [pos[0],pos[1],1])
            pos = np.dot(self.inverse_remote_homography, pos)
            pos = pos[:2]/pos[2]

            if event.LeftDown():
                button = 0
            elif event.RightDown():
                button = 1
            else:
                button = None

            self.process_clicked_point(pos, button)
        # self.invalidate_cache()

    def process_clicked_point(self, pos, button):
        """
        :param pos: Raw image coordinates that were clicked.
        :type pos: 2-array

        :param button: The mouse button that was clicked (0 for left, 1 for
            right)
        :type button: 0 | 1

        """
        pass

    def generate_histogram(self):
        """Generate histogram for current image.

        """
        panel_width, panel_height = self.wx_histogram_panel.GetSize()

        if self.raw_image.dtype == np.uint8:
            v = np.histogram(self.raw_image, np.linspace(-0.5, 255.5, 257))[0]
        elif self.raw_image.dtype == np.uint16:
            v = np.histogram(self.raw_image, np.linspace(0, 65535, 257))[0]
        else:
            raise Exception(self.raw_image.dtype)

        vmax2, vmax1 = np.sort(v)[-2:]
        #vmax = min([vmax1,vmax2*2])
        vmax = vmax1
        v = v/(vmax)*panel_height
        h = panel_height - np.round(v).astype(np.int)
        image = np.full((panel_height,256,3), 255, np.uint8)

        for i in range(len(h)):
            image[h[i]:,i,1:] = 0

        self._histogram = image

    def on_paint(self, event=None):
        """Called on event wx.EVT_PAINT.

        """
        if self.wx_bitmap is not None:
            pdc = wx.PaintDC(self.wx_panel)
            dc = wx.GCDC(pdc)

            panel_width, panel_height = self.wx_panel.GetSize()
            dx = (panel_width - self.panel_image_width)//2
            dy = (panel_height - self.panel_image_height)//2
            dc.DrawBitmap(self.wx_bitmap, dx, dy)

        if self._histogram is not None:
            panel_width, panel_height = self.wx_histogram_panel.GetSize()
            image = cv2.resize(self._histogram, dsize=(panel_width,
                               panel_height), interpolation=cv2.INTER_NEAREST)

            wx_image = wx.EmptyImage(panel_width, panel_height)
            wx_image.SetData(image.tostring())
            wx_histogram_bitmap = wx_image.ConvertToBitmap()

            pdc = wx.PaintDC(self.wx_histogram_panel)
            dc = wx.GCDC(pdc)
            dc.DrawBitmap(wx_histogram_bitmap, 0, 0)

        if event is not None:
            event.Skip()

    def refresh(self, event):
        """Useful to bind the Refresh of self.wx_panel to an event.

        """
        event.Skip()
        self.wx_panel.Refresh(True)

    def update_status_msg(self, string):
        if self.status_static_text is None: return None
        if self._stop: return None  # Check for a request to stop.
        string0 = self.status_static_text.GetLabel()
        self.status_static_text.SetLabel(string)
        if len(string) > 0 and string[0] == u"☒":
            self.status_static_text.SetForegroundColour(BRIGHT_RED)
        else:
            self.status_static_text.SetForegroundColour((0,0,0))

        if len(string0) != len(string):
            self.status_static_text.GetParent().Layout()

    def release(self):
        # Set the _stop flag so that any threads can check and stop before
        # trying to access.
        self._stop = True

        self.wx_panel.Unbind(wx.EVT_PAINT)
        self.wx_panel.Unbind(wx.EVT_SIZE)

        if self.update_image_thread:
            self.update_image_thread.stop()
