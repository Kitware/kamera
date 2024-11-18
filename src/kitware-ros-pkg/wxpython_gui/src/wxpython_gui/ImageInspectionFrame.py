import wx
import wxpython_gui.system_control_panel.form_builder_output_imagery_inspection as form_builder_output_imagery_inspection
from wxpython_gui.FullAndZoomView import FullAndZoomView
from wxpython_gui.cfg import SYS_CFG


class ImageInspectionFrame(form_builder_output_imagery_inspection.MainFrame):
    """Provides a zoomed in and full-frame view of the camera selected.

    """
    def __init__(self, parent, topic_names,
                 stream=None, compressed=False):
        # type: (Any, dict, Callable, Any, bool) -> None
        """
        :param parent: Parent.
        :type parent: wx object

        :param effort_metadata_dict: Dictionary with key being the effort
            nickname and the value being a dictionary with keys 'project_name',
            'aircraft', 'flight', and 'field_notes'.
        :type effort_metadata_dict: dict

        :param effort_combo_box: Combo box for the collection event selection.
        :type effort_combo_box: wx.ComboBox

        :param edit_effort_name: Name of existing event that we want to edit.
        :type edit_effort_name: str

        """
        # Initialize parent class
        self.wx_block = form_builder_output_imagery_inspection.MainFrame.__init__(self, parent)

        self.topic_names = topic_names
        self.full_view_rp = None
        self.compressed = compressed
        self.parent = parent

        ir_contrast_strength = SYS_CFG["ir_contrast_strength"]
        self.ir_contrast_strength_txt_ctrl.SetValue(str(ir_contrast_strength))

        for i in range(self.image_stream_combo_box.GetCount()):
            if stream == self.image_stream_combo_box.GetString(i):
                self.image_stream_combo_box.SetSelection(i)
                wx.CallAfter(self.on_select_stream, None)

        self.Bind(wx.EVT_CLOSE, self.when_closed)

    def on_select_stream(self, event=None):
        if self.full_view_rp is not None:
            self.full_view_rp.release()

        ind = self.image_stream_combo_box.GetCurrentSelection()

        if ind == 0:
            srv_topic = self.topic_names['left_rgb_srv_topic']
        elif ind == 1:
            srv_topic = self.topic_names['center_rgb_srv_topic']
        elif ind == 2:
            srv_topic = self.topic_names['right_rgb_srv_topic']
        elif ind == 3:
            srv_topic = self.topic_names['left_ir_srv_topic']
        elif ind == 4:
            srv_topic = self.topic_names['center_ir_srv_topic']
        elif ind == 5:
            srv_topic = self.topic_names['right_ir_srv_topic']
        elif ind == 6:
            srv_topic = self.topic_names['left_uv_srv_topic']
        elif ind == 7:
            srv_topic = self.topic_names['center_uv_srv_topic']
        elif ind == 8:
            srv_topic = self.topic_names['right_uv_srv_topic']

        self.full_view_rp = FullAndZoomView(self.full_view_panel,
                                            self.zoomed_view_panel,
                                            self.histogram_panel,
                                            self.zoom_slider,
                                            self.status_text, srv_topic,
                                            self.parent, self.compressed)

    def on_toggle_saturated_pixels(self, event):
        """Toggle coloring of saturated pixels red.

        """
        SYS_CFG["show_saturated_pixels"] = not SYS_CFG["show_saturated_pixels"]

    def on_ir_contrast_strength(self, event=None):
        try:
            SYS_CFG["ir_contrast_strength"] = float(
                    self.ir_contrast_strength_txt_ctrl.GetValue())
        except ValueError:
            pass

    def on_close_button(self, event=None):
        """When the 'Cancel' button is selected.

        """
        self.Close()

    def when_closed(self, event=None):
        if self.full_view_rp is not None:
            self.full_view_rp.release()
        event.Skip()

