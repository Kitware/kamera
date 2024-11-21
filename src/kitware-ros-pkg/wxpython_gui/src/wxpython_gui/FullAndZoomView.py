import wx
from wxpython_gui.RemoteImagePanelZoom import RemoteImagePanelZoom
from wxpython_gui.RemoteImagePanelFit import RemoteImagePanelFit


class FullAndZoomView(object):
    def __init__(
        self,
        full_view_panel,
        zoomed_view_panel,
        histogram_panel,
        zoom_slider,
        status_static_text,
        srv_topic,
        parent,
        compressed=False,
    ):
        """
        :param wx_panel: Panel to add the image to.
        :type wx_panel: wx.Panel

        """
        self.zoom_panel = RemoteImagePanelZoom(
            wx_panel=zoomed_view_panel,
            srv_topic=srv_topic,
            zoom_slider=zoom_slider,
            compressed=compressed,
        )

        self.fit_panel = RemoteImagePanelFit(
            wx_panel=full_view_panel,
            srv_topic=srv_topic,
            click_callback=self.zoom_panel.process_clicked_point,
            status_static_text=status_static_text,
            compressed=compressed,
            wx_histogram_panel=histogram_panel,
        )
        # self.zoom_panel.start_image_thread()
        if "rgb" in srv_topic:
            self.zoom_enabled = False
            parent._image_inspection_frame.m_panel37.Hide()
            parent._image_inspection_frame.cueing_left_image_title3.Hide()
        else:
            self.zoom_enabled = True
            self.zoom_panel.start_image_thread()
            self.fit_panel.start_image_thread()

    def set_zoom(self, zoom):
        """
        :param zoom: Zoom percentage.
        :type zoom: float
        """
        # if not zoom_enabled:
        #    return
        self._zoom = zoom
        self.zoom_label.SetLabel("{}%".format(int(np.round(zoom))))
        self.zoom_panel.update_all()

    def set_center(self, center):
        """
        :param center: Location for the zoom center in the original image's coordinates.
        :type center: 2-array
        """
        # if not zoom_enabled:
        #    return
        self._center = center
        self.zoom_panel.update_all()

    def release(self):
        self.fit_panel.release()
        # if self.zoom_enabled:
        #    self.zoom_panel.release()
