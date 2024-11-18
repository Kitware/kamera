import wx
import wxpython_gui.system_control_panel.form_builder_output_log_panel as form_builder_output_log_panel


class LogFrame(form_builder_output_log_panel.MainFrame):
    """.

    """
    def __init__(self, parent):
        """

        """
        # Initialize parent class
        form_builder_output_log_panel.MainFrame.__init__(self, parent)

        bsizer = wx.BoxSizer(wx.VERTICAL)
        self.static_text = wx.StaticText(parent, wx.ID_ANY,
                                         'Log Messages', wx.DefaultPosition,
                                         wx.DefaultSize,
                                         wx.ALIGN_CENTRE|wx.ST_NO_AUTORESIZE)

        self.static_text.Wrap( -1 )
        self.static_text.SetFont(wx.Font(14, 70, 90, 92, False,
                                         wx.EmptyString))
        bsizer.Add(self.static_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        flags = wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER|wx.VSCROLL

        self.list_ctrl = wx.ListCtrl(self, wx.ID_ANY, wx.DefaultPosition,
                                     wx.Size( -1,-1), flags)
        bsizer.Add(self.list_ctrl, 1, wx.ALL|wx.EXPAND, 5)

        self.message_panel.SetSizer(bsizer)
        self.message_panel.Layout()
        bsizer.Fit(self.message_panel)

        self.list_ctrl.InsertColumn(0, 'Timestamp', wx.LIST_FORMAT_CENTER,
                                    width=250)
        self.list_ctrl.InsertColumn(1, 'Type', wx.LIST_FORMAT_CENTER, width=50)
        self.list_ctrl.InsertColumn(2, 'Message', wx.LIST_FORMAT_LEFT,
                                    width=1100)

        self.Show()
        self.SetMinSize(self.GetSize())

    def add_message(self, msg_type, msg):
        tstamp = str(datetime.datetime.utcfromtimestamp(time.time()))
        #i = self.list_ctrl.GetItemCount()
        i = 0
        self.list_ctrl.InsertStringItem(i, tstamp)
        self.list_ctrl.SetStringItem(i, 1, msg_type)
        self.list_ctrl.SetStringItem(i, 2, msg)

    def on_cancel(self, event=None):
        """when the 'cancel' button is selected.

        """
        self.close()
