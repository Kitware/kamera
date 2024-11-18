import wx
import wxpython_gui.system_control_panel.form_builder_output_hot_key_list as form_builder_output_hot_key_list


class HotKeyList(form_builder_output_hot_key_list.MainFrame):
    def __init__(self, parent):
        """

        """
        # Initialize parent class
        form_builder_output_hot_key_list.MainFrame.__init__(self, parent)

        # ----------------------------- Hot Keys -----------------------------
        entries = [wx.AcceleratorEntry() for _ in range(1)]

        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.on_cancel, id=random_id)
        entries[0].Set(wx.ACCEL_NORMAL, wx.WXK_ESCAPE, random_id)

        accel = wx.AcceleratorTable(entries)
        self.SetAcceleratorTable(accel)
        # --------------------------------------------------------------------

        self.Show()
        self.SetMinSize(self.GetSize())

    def on_cancel(self, event=None):
        """When the 'Cancel' button is selected.

        """
        self.Close()

