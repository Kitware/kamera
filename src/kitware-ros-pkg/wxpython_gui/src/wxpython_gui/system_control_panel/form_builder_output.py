# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Jan 30 2023)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

ID_START_DETECTOR_SYS0_CENTER = 1000
ID_START_DETECTOR_SYS1_LEFT = 1001
ID_START_DETECTOR_SYS2_RIGHT = 1002
ID_STOP_DETECTOR_SYS0_CENTER = 1003
ID_STOP_DETECTOR_SYS1_LEFT = 1004
ID_STOP_DETECTOR_SYS2_RIGHT = 1005
wx._ID_ANY = 1006

###########################################################################
## Class MainFrame
###########################################################################

class MainFrame ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"System Control Panel", pos = wx.DefaultPosition, size = wx.Size( 1117,1062 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

        self.SetSizeHintsSz( wx.Size( 400,400 ), wx.DefaultSize )
        self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
        self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )

        main_size = wx.BoxSizer( wx.HORIZONTAL )

        bSizer20 = wx.BoxSizer( wx.VERTICAL )

        self.ins_control_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bSizer191 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText142 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Navigation Data", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText142.Wrap( -1 )
        self.m_staticText142.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

        bSizer191.Add( self.m_staticText142, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.RIGHT|wx.LEFT, 5 )

        self.m_staticline4 = wx.StaticLine( self.ins_control_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer191.Add( self.m_staticline4, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        bSizer181 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText181 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Lat (deg)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText181.Wrap( -1 )
        self.m_staticText181.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer181.Add( self.m_staticText181, 0, wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )

        self.lat_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        self.lat_txtctrl.SetMinSize( wx.Size( 180,-1 ) )

        bSizer181.Add( self.lat_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer181, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 5 )

        bSizer1811 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Lon (deg)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText.Wrap( -1 )
        self.m_staticText.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer1811.Add( self.m_staticText, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        self.lon_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer1811.Add( self.lon_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer1811, 0, wx.EXPAND, 5 )

        bSizer1812 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText1812 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Alt HAE (m)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1812.Wrap( -1 )
        self.m_staticText1812.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer1812.Add( self.m_staticText1812, 0, wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )

        self.alt_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer1812.Add( self.alt_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer1812, 0, wx.EXPAND, 5 )

        bSizer18121 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText18121 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Alt MSL (m)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText18121.Wrap( -1 )
        self.m_staticText18121.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer18121.Add( self.m_staticText18121, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.alt_msl_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer18121.Add( self.alt_msl_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer18121, 1, wx.EXPAND, 5 )

        bSizer18191 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText18191 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Speed (kts)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText18191.Wrap( -1 )
        self.m_staticText18191.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer18191.Add( self.m_staticText18191, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        self.speed_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer18191.Add( self.speed_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer18191, 1, wx.EXPAND, 5 )

        bSizer1819 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText1819 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Heading (deg)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1819.Wrap( -1 )
        self.m_staticText1819.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer1819.Add( self.m_staticText1819, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        self.heading_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer1819.Add( self.heading_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer1819, 0, wx.EXPAND, 5 )

        bSizer1818 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText1818 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Pitch (deg)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1818.Wrap( -1 )
        self.m_staticText1818.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer1818.Add( self.m_staticText1818, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.pitch_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer1818.Add( self.pitch_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer1818, 0, wx.EXPAND, 5 )

        bSizer1817 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText1817 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Roll (deg)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1817.Wrap( -1 )
        self.m_staticText1817.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer1817.Add( self.m_staticText1817, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.roll_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer1817.Add( self.roll_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer1817, 0, wx.EXPAND, 5 )

        bSizer1817131 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText1817131 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Time (s)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1817131.Wrap( -1 )
        self.m_staticText1817131.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer1817131.Add( self.m_staticText1817131, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        self.ins_time_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTRE|wx.TE_READONLY )
        bSizer1817131.Add( self.ins_time_txtctrl, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer191.Add( bSizer1817131, 0, wx.EXPAND, 5 )

        bSizer18171313 = wx.BoxSizer( wx.HORIZONTAL )

        bSizer511 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText181713131 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"GNSS Status", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText181713131.Wrap( -1 )
        self.m_staticText181713131.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer511.Add( self.m_staticText181713131, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.RIGHT|wx.LEFT, 5 )

        self.gnss_status_flag_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, u"not available", wx.DefaultPosition, wx.Size( -1,-1 ), wx.TE_CENTRE|wx.TE_READONLY )
        bSizer511.Add( self.gnss_status_flag_txtctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )


        bSizer18171313.Add( bSizer511, 1, wx.EXPAND, 5 )

        bSizer51 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText18171313 = wx.StaticText( self.ins_control_panel, wx.ID_ANY, u"Align Status", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText18171313.Wrap( -1 )
        self.m_staticText18171313.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer51.Add( self.m_staticText18171313, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.RIGHT|wx.LEFT, 5 )

        self.ins_status_flag_txtctrl = wx.TextCtrl( self.ins_control_panel, wx.ID_ANY, u"unknown", wx.DefaultPosition, wx.Size( 120,-1 ), wx.TE_CENTRE|wx.TE_READONLY )
        bSizer51.Add( self.ins_status_flag_txtctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )


        bSizer18171313.Add( bSizer51, 0, wx.EXPAND, 5 )


        bSizer191.Add( bSizer18171313, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5 )


        self.ins_control_panel.SetSizer( bSizer191 )
        self.ins_control_panel.Layout()
        bSizer191.Fit( self.ins_control_panel )
        bSizer20.Add( self.ins_control_panel, 0, wx.EXPAND|wx.BOTTOM, 5 )

        self.camera_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        m_staticText14211 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText142111 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Camera Settings", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText142111.Wrap( -1 )
        self.m_staticText142111.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

        m_staticText14211.Add( self.m_staticText142111, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.RIGHT|wx.LEFT, 5 )

        bSizer55 = wx.BoxSizer( wx.HORIZONTAL )

        camera_setting_rgb_uv_comboChoices = [ u"RGB", u"IR", u"UV" ]
        self.camera_setting_rgb_uv_combo = wx.ComboBox( self.camera_panel, wx.ID_ANY, u"RGB", wx.DefaultPosition, wx.DefaultSize, camera_setting_rgb_uv_comboChoices, wx.CB_READONLY )
        self.camera_setting_rgb_uv_combo.SetSelection( 0 )
        bSizer55.Add( self.camera_setting_rgb_uv_combo, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        camera_setting_subsysChoices = [ u"Left", u"Center", u"Right", u"All" ]
        self.camera_setting_subsys = wx.ComboBox( self.camera_panel, wx.ID_ANY, u"Right", wx.DefaultPosition, wx.DefaultSize, camera_setting_subsysChoices, wx.CB_READONLY )
        self.camera_setting_subsys.SetSelection( 3 )
        bSizer55.Add( self.camera_setting_subsys, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )


        m_staticText14211.Add( bSizer55, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_staticline5 = wx.StaticLine( self.camera_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        m_staticText14211.Add( self.m_staticline5, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        self.m_staticText42 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Auto Exposure (ms)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText42.Wrap( -1 )
        self.m_staticText42.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        m_staticText14211.Add( self.m_staticText42, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer442 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText423 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Min:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText423.Wrap( -1 )
        self.m_staticText423.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer442.Add( self.m_staticText423, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 5 )

        self.exposure_min_value_txt_ctrl = wx.TextCtrl( self.camera_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 70,-1 ), wx.TE_CENTRE )
        bSizer442.Add( self.exposure_min_value_txt_ctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )


        bSizer442.AddSpacer( ( 10, 0), 1, wx.EXPAND, 5 )

        self.m_staticText4231 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Max:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4231.Wrap( -1 )
        self.m_staticText4231.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer442.Add( self.m_staticText4231, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 5 )

        self.exposure_max_value_txt_ctrl = wx.TextCtrl( self.camera_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 70,-1 ), wx.TE_CENTRE )
        bSizer442.Add( self.exposure_max_value_txt_ctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )


        m_staticText14211.Add( bSizer442, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_staticline51 = wx.StaticLine( self.camera_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        m_staticText14211.Add( self.m_staticline51, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        self.m_staticText422 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Auto Gain (0-32)", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText422.Wrap( -1 )
        self.m_staticText422.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        m_staticText14211.Add( self.m_staticText422, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL|wx.RIGHT|wx.LEFT, 5 )

        bSizer4421 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText4232 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Min:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4232.Wrap( -1 )
        self.m_staticText4232.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer4421.Add( self.m_staticText4232, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 5 )

        self.gain_min_value_txt_ctrl = wx.TextCtrl( self.camera_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 70,-1 ), wx.TE_CENTRE )
        bSizer4421.Add( self.gain_min_value_txt_ctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )


        bSizer4421.AddSpacer( ( 10, 0), 1, wx.EXPAND, 5 )

        self.m_staticText42311 = wx.StaticText( self.camera_panel, wx.ID_ANY, u"Max:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText42311.Wrap( -1 )
        self.m_staticText42311.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer4421.Add( self.m_staticText42311, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 5 )

        self.gain_max_value_txt_ctrl = wx.TextCtrl( self.camera_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 70,-1 ), wx.TE_CENTRE )
        bSizer4421.Add( self.gain_max_value_txt_ctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )


        m_staticText14211.Add( bSizer4421, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_staticline52 = wx.StaticLine( self.camera_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        m_staticText14211.Add( self.m_staticline52, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        bSizer44211 = wx.BoxSizer( wx.HORIZONTAL )

        self.txtNUC = wx.StaticText( self.camera_panel, wx.ID_ANY, u"IR NUC Time (min):", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.txtNUC.Wrap( -1 )
        self.txtNUC.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        bSizer44211.Add( self.txtNUC, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT|wx.RIGHT, 5 )

        self.ir_nuc_time = wx.TextCtrl( self.camera_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 80,-1 ), wx.TE_CENTRE )
        bSizer44211.Add( self.ir_nuc_time, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )


        m_staticText14211.Add( bSizer44211, 1, wx.EXPAND, 5 )

        bSizer442111 = wx.BoxSizer( wx.HORIZONTAL )


        m_staticText14211.Add( bSizer442111, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.BOTTOM, 5 )

        self.m_button10 = wx.Button( self.camera_panel, wx.ID_ANY, u"Set Camera Parameter", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_button10.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        m_staticText14211.Add( self.m_button10, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.RIGHT|wx.LEFT, 5 )

        self.m_manual_ir_nuc = wx.Button( self.camera_panel, wx.ID_ANY, u"Manual IR NUC", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_manual_ir_nuc.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

        m_staticText14211.Add( self.m_manual_ir_nuc, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.RIGHT|wx.LEFT, 5 )


        self.camera_panel.SetSizer( m_staticText14211 )
        self.camera_panel.Layout()
        m_staticText14211.Fit( self.camera_panel )
        bSizer20.Add( self.camera_panel, 0, wx.EXPAND|wx.BOTTOM, 5 )

        self.flight_data_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bSizer391 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText14211 = wx.StaticText( self.flight_data_panel, wx.ID_ANY, u"Data Collection", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText14211.Wrap( -1 )
        self.m_staticText14211.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

        bSizer391.Add( self.m_staticText14211, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.RIGHT|wx.LEFT, 5 )

        self.m_staticline1 = wx.StaticLine( self.flight_data_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer391.Add( self.m_staticline1, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        bSizer41 = wx.BoxSizer( wx.HORIZONTAL )

        bSizer42 = wx.BoxSizer( wx.VERTICAL )

        bSizer441 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText18171311 = wx.StaticText( self.flight_data_panel, wx.ID_ANY, u"Effort", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText18171311.Wrap( -1 )
        self.m_staticText18171311.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bSizer441.Add( self.m_staticText18171311, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        effort_combo_boxChoices = []
        self.effort_combo_box = wx.ComboBox( self.flight_data_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, effort_combo_boxChoices, 0 )
        bSizer441.Add( self.effort_combo_box, 1, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND|wx.ALL, 5 )


        bSizer42.Add( bSizer441, 1, wx.EXPAND, 5 )

        bSizer43 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_button4 = wx.Button( self.flight_data_panel, wx.ID_ANY, u"New", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer43.Add( self.m_button4, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_button5 = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Edit", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer43.Add( self.m_button5, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_button6 = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Delete", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer43.Add( self.m_button6, 0, wx.ALL, 5 )


        bSizer42.Add( bSizer43, 1, wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_staticline41 = wx.StaticLine( self.flight_data_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer42.Add( self.m_staticline41, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        bSizer45 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText33 = wx.StaticText( self.flight_data_panel, wx.ID_ANY, u"Flight:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText33.Wrap( -1 )
        self.m_staticText33.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bSizer45.Add( self.m_staticText33, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5 )

        self.m_staticText34 = wx.StaticText( self.flight_data_panel, wx.ID_ANY, u"FL", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText34.Wrap( -1 )
        bSizer45.Add( self.m_staticText34, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.flight_number_text_ctrl = wx.TextCtrl( self.flight_data_panel, wx.ID_ANY, u"00", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer45.Add( self.flight_number_text_ctrl, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5 )


        bSizer42.Add( bSizer45, 1, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 5 )

        bSizer451 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText331 = wx.StaticText( self.flight_data_panel, wx.ID_ANY, u"Observer", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText331.Wrap( -1 )
        self.m_staticText331.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bSizer451.Add( self.m_staticText331, 0, wx.RIGHT|wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.observer_text_ctrl = wx.TextCtrl( self.flight_data_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer451.Add( self.observer_text_ctrl, 1, wx.RIGHT|wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5 )


        bSizer42.Add( bSizer451, 1, wx.EXPAND, 5 )


        bSizer41.Add( bSizer42, 1, wx.EXPAND, 5 )


        bSizer391.Add( bSizer41, 0, wx.EXPAND, 5 )

        bSizer401 = wx.BoxSizer( wx.HORIZONTAL )


        bSizer391.Add( bSizer401, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_staticline3 = wx.StaticLine( self.flight_data_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer391.Add( self.m_staticline3, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        bSizer611 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_button9 = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Add Note to Log", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer611.Add( self.m_button9, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_button81 = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Set Collection Mode", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer611.Add( self.m_button81, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        bSizer391.Add( bSizer611, 0, wx.EXPAND, 5 )

        bSizer44 = wx.BoxSizer( wx.HORIZONTAL )

        self.recording_gauge = wx.Gauge( self.flight_data_panel, wx.ID_ANY, 1, wx.DefaultPosition, wx.Size( 25,25 ), wx.GA_HORIZONTAL )
        self.recording_gauge.SetValue( 0 )
        bSizer44.Add( self.recording_gauge, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.start_collecting_button = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Start Collecting", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer44.Add( self.start_collecting_button, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_button8 = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Stop Collecting", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer44.Add( self.m_button8, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


        bSizer391.Add( bSizer44, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.m_staticline411 = wx.StaticLine( self.flight_data_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer391.Add( self.m_staticline411, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )

        bSizer443 = wx.BoxSizer( wx.HORIZONTAL )

        self.detectors_gauge = wx.Gauge( self.flight_data_panel, wx.ID_ANY, 1, wx.DefaultPosition, wx.Size( 25,25 ), wx.GA_HORIZONTAL )
        self.detectors_gauge.SetValue( 0 )
        bSizer443.Add( self.detectors_gauge, 0, wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )

        self.start_detectors_button = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Start Detectors", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer443.Add( self.start_detectors_button, 0, wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )

        self.stop_detectors_button = wx.Button( self.flight_data_panel, wx.ID_ANY, u"Stop Detectors", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer443.Add( self.stop_detectors_button, 0, wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5 )


        bSizer391.Add( bSizer443, 1, wx.EXPAND, 5 )

        self.m_staticline13 = wx.StaticLine( self.flight_data_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer391.Add( self.m_staticline13, 0, wx.EXPAND|wx.RIGHT|wx.LEFT, 5 )

        self.nas_disk_space = wx.StaticText( self.flight_data_panel, wx.ID_ANY, u"NAS Disk Space: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.nas_disk_space.Wrap( -1 )
        self.nas_disk_space.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bSizer391.Add( self.nas_disk_space, 0, wx.TOP|wx.RIGHT|wx.LEFT, 5 )


        bSizer391.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )


        self.flight_data_panel.SetSizer( bSizer391 )
        self.flight_data_panel.Layout()
        bSizer391.Fit( self.flight_data_panel )
        bSizer20.Add( self.flight_data_panel, 1, wx.EXPAND, 5 )

        self.m_panel7 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bSizer17 = wx.BoxSizer( wx.VERTICAL )

        self.close_button = wx.Button( self.m_panel7, wx.ID_ANY, u"Close", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
        bSizer17.Add( self.close_button, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel7.SetSizer( bSizer17 )
        self.m_panel7.Layout()
        bSizer17.Fit( self.m_panel7 )
        bSizer20.Add( self.m_panel7, 0, wx.EXPAND|wx.TOP, 5 )


        main_size.Add( bSizer20, 0, wx.ALIGN_CENTER_VERTICAL|wx.EXPAND|wx.ALL, 5 )

        bsizer12 = wx.BoxSizer( wx.VERTICAL )

        self.m_panel37 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bSizer59 = wx.BoxSizer( wx.VERTICAL )

        bSizer61 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText40 = wx.StaticText( self.m_panel37, wx.ID_ANY, u"Camera/Mount Configuration", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText40.Wrap( -1 )
        self.m_staticText40.SetFont( wx.Font( 12, 70, 90, 92, False, wx.EmptyString ) )

        bSizer61.Add( self.m_staticText40, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        camera_config_comboChoices = []
        self.camera_config_combo = wx.ComboBox( self.m_panel37, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 400,-1 ), camera_config_comboChoices, 0 )
        bSizer61.Add( self.camera_config_combo, 0, wx.ALL, 5 )


        bSizer61.AddSpacer( ( 20, 0), 1, wx.EXPAND, 5 )


        bSizer59.Add( bSizer61, 1, wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel37.SetSizer( bSizer59 )
        self.m_panel37.Layout()
        bSizer59.Fit( self.m_panel37 )
        bsizer12.Add( self.m_panel37, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND|wx.TOP|wx.RIGHT, 5 )

        self.images_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,-1 ), wx.TAB_TRAVERSAL )
        self.images_panel.SetFont( wx.Font( 9, 70, 90, 90, False, wx.EmptyString ) )

        bSizer16 = wx.BoxSizer( wx.VERTICAL )

        top_row_bSizer = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel_left_rgb = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        left_bsizer0 = wx.BoxSizer( wx.VERTICAL )

        self.cueing_left_image_title3 = wx.StaticText( self.m_panel_left_rgb, wx.ID_ANY, u"Left RGB", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.cueing_left_image_title3.Wrap( -1 )
        self.cueing_left_image_title3.SetFont( wx.Font( 14, 74, 90, 92, False, "Sans" ) )

        left_bsizer0.Add( self.cueing_left_image_title3, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        self.left_rgb_panel = wx.Panel( self.m_panel_left_rgb, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        left_bsizer0.Add( self.left_rgb_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.left_rgb_histogram_panel = wx.Panel( self.m_panel_left_rgb, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        left_bsizer0.Add( self.left_rgb_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )

        self.left_rgb_status_text = wx.StaticText( self.m_panel_left_rgb, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.left_rgb_status_text.Wrap( -1 )
        self.left_rgb_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        left_bsizer0.Add( self.left_rgb_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_left_rgb.SetSizer( left_bsizer0 )
        self.m_panel_left_rgb.Layout()
        left_bsizer0.Fit( self.m_panel_left_rgb )
        top_row_bSizer.Add( self.m_panel_left_rgb, 1, wx.EXPAND, 5 )

        self.m_panel_center_rgb = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        right_bsizer0 = wx.BoxSizer( wx.VERTICAL )

        self.cueing_right_image_title = wx.StaticText( self.m_panel_center_rgb, wx.ID_ANY, u"Center RGB", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.cueing_right_image_title.Wrap( -1 )
        self.cueing_right_image_title.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer0.Add( self.cueing_right_image_title, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer32 = wx.BoxSizer( wx.VERTICAL )

        self.center_rgb_panel = wx.Panel( self.m_panel_center_rgb, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer32.Add( self.center_rgb_panel, 5, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND|wx.TOP|wx.RIGHT|wx.LEFT, 5 )

        self.center_rgb_histogram_panel = wx.Panel( self.m_panel_center_rgb, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer32.Add( self.center_rgb_histogram_panel, 1, wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        right_bsizer0.Add( bSizer32, 1, wx.EXPAND, 5 )

        self.center_rgb_status_text = wx.StaticText( self.m_panel_center_rgb, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.center_rgb_status_text.Wrap( -1 )
        self.center_rgb_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer0.Add( self.center_rgb_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_center_rgb.SetSizer( right_bsizer0 )
        self.m_panel_center_rgb.Layout()
        right_bsizer0.Fit( self.m_panel_center_rgb )
        top_row_bSizer.Add( self.m_panel_center_rgb, 1, wx.EXPAND, 5 )

        self.m_panel_right_rgb = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        right_bsizer0 = wx.BoxSizer( wx.VERTICAL )

        self.ptz_image_title = wx.StaticText( self.m_panel_right_rgb, wx.ID_ANY, u"Right RGB", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.ptz_image_title.Wrap( -1 )
        self.ptz_image_title.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer0.Add( self.ptz_image_title, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer38 = wx.BoxSizer( wx.VERTICAL )

        self.right_rgb_panel = wx.Panel( self.m_panel_right_rgb, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer38.Add( self.right_rgb_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.right_rgb_histogram_panel = wx.Panel( self.m_panel_right_rgb, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer38.Add( self.right_rgb_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        right_bsizer0.Add( bSizer38, 1, wx.EXPAND, 5 )

        self.right_rgb_status_text = wx.StaticText( self.m_panel_right_rgb, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.right_rgb_status_text.Wrap( -1 )
        self.right_rgb_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer0.Add( self.right_rgb_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_right_rgb.SetSizer( right_bsizer0 )
        self.m_panel_right_rgb.Layout()
        right_bsizer0.Fit( self.m_panel_right_rgb )
        top_row_bSizer.Add( self.m_panel_right_rgb, 1, wx.EXPAND, 5 )


        bSizer16.Add( top_row_bSizer, 1, wx.EXPAND, 5 )

        middle_row_bSizer = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel_left_ir = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        left_bsizer1 = wx.BoxSizer( wx.VERTICAL )

        self.cueing_left_image_title1 = wx.StaticText( self.m_panel_left_ir, wx.ID_ANY, u"Left IR", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.cueing_left_image_title1.Wrap( -1 )
        self.cueing_left_image_title1.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        left_bsizer1.Add( self.cueing_left_image_title1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer36 = wx.BoxSizer( wx.VERTICAL )

        self.left_ir_panel = wx.Panel( self.m_panel_left_ir, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer36.Add( self.left_ir_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.left_ir_histogram_panel = wx.Panel( self.m_panel_left_ir, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer36.Add( self.left_ir_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        left_bsizer1.Add( bSizer36, 1, wx.EXPAND, 5 )

        self.left_ir_status_text = wx.StaticText( self.m_panel_left_ir, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.left_ir_status_text.Wrap( -1 )
        self.left_ir_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        left_bsizer1.Add( self.left_ir_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_left_ir.SetSizer( left_bsizer1 )
        self.m_panel_left_ir.Layout()
        left_bsizer1.Fit( self.m_panel_left_ir )
        middle_row_bSizer.Add( self.m_panel_left_ir, 1, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_panel_center_ir = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        right_bsizer1 = wx.BoxSizer( wx.VERTICAL )

        self.cueing_right_image_title1 = wx.StaticText( self.m_panel_center_ir, wx.ID_ANY, u"Center IR", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.cueing_right_image_title1.Wrap( -1 )
        self.cueing_right_image_title1.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer1.Add( self.cueing_right_image_title1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer33 = wx.BoxSizer( wx.VERTICAL )

        self.center_ir_panel = wx.Panel( self.m_panel_center_ir, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer33.Add( self.center_ir_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.center_ir_histogram_panel = wx.Panel( self.m_panel_center_ir, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer33.Add( self.center_ir_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        right_bsizer1.Add( bSizer33, 1, wx.EXPAND, 5 )

        self.center_ir_status_text = wx.StaticText( self.m_panel_center_ir, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.center_ir_status_text.Wrap( -1 )
        self.center_ir_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer1.Add( self.center_ir_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_center_ir.SetSizer( right_bsizer1 )
        self.m_panel_center_ir.Layout()
        right_bsizer1.Fit( self.m_panel_center_ir )
        middle_row_bSizer.Add( self.m_panel_center_ir, 1, wx.EXPAND, 5 )

        self.m_panel_right_ir = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        ptz_bsizer1 = wx.BoxSizer( wx.VERTICAL )

        self.ptz_image_title1 = wx.StaticText( self.m_panel_right_ir, wx.ID_ANY, u"Right IR", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.ptz_image_title1.Wrap( -1 )
        self.ptz_image_title1.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        ptz_bsizer1.Add( self.ptz_image_title1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer39 = wx.BoxSizer( wx.VERTICAL )

        self.right_ir_panel = wx.Panel( self.m_panel_right_ir, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer39.Add( self.right_ir_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.right_ir_histogram_panel = wx.Panel( self.m_panel_right_ir, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer39.Add( self.right_ir_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        ptz_bsizer1.Add( bSizer39, 1, wx.EXPAND, 5 )

        self.right_ir_status_text = wx.StaticText( self.m_panel_right_ir, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.right_ir_status_text.Wrap( -1 )
        self.right_ir_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        ptz_bsizer1.Add( self.right_ir_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_right_ir.SetSizer( ptz_bsizer1 )
        self.m_panel_right_ir.Layout()
        ptz_bsizer1.Fit( self.m_panel_right_ir )
        middle_row_bSizer.Add( self.m_panel_right_ir, 1, wx.EXPAND, 5 )


        bSizer16.Add( middle_row_bSizer, 1, wx.EXPAND, 5 )

        bottom_row_bSizer = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel_left_uv = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        left_bsizer2 = wx.BoxSizer( wx.VERTICAL )

        self.cueing_left_image_title2 = wx.StaticText( self.m_panel_left_uv, wx.ID_ANY, u"Left UV", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.cueing_left_image_title2.Wrap( -1 )
        self.cueing_left_image_title2.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        left_bsizer2.Add( self.cueing_left_image_title2, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer37 = wx.BoxSizer( wx.VERTICAL )

        self.left_uv_panel = wx.Panel( self.m_panel_left_uv, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer37.Add( self.left_uv_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.left_uv_histogram_panel = wx.Panel( self.m_panel_left_uv, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer37.Add( self.left_uv_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        left_bsizer2.Add( bSizer37, 1, wx.EXPAND, 5 )

        self.left_uv_status_text = wx.StaticText( self.m_panel_left_uv, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.left_uv_status_text.Wrap( -1 )
        self.left_uv_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        left_bsizer2.Add( self.left_uv_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_left_uv.SetSizer( left_bsizer2 )
        self.m_panel_left_uv.Layout()
        left_bsizer2.Fit( self.m_panel_left_uv )
        bottom_row_bSizer.Add( self.m_panel_left_uv, 1, wx.EXPAND, 5 )

        self.m_panel_center_uv = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        right_bsizer2 = wx.BoxSizer( wx.VERTICAL )

        self.cueing_right_image_title2 = wx.StaticText( self.m_panel_center_uv, wx.ID_ANY, u"Center UV", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.cueing_right_image_title2.Wrap( -1 )
        self.cueing_right_image_title2.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer2.Add( self.cueing_right_image_title2, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer34 = wx.BoxSizer( wx.VERTICAL )

        self.center_uv_panel = wx.Panel( self.m_panel_center_uv, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer34.Add( self.center_uv_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.center_uv_histogram_panel = wx.Panel( self.m_panel_center_uv, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer34.Add( self.center_uv_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        right_bsizer2.Add( bSizer34, 1, wx.EXPAND, 5 )

        self.center_uv_status_text = wx.StaticText( self.m_panel_center_uv, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.center_uv_status_text.Wrap( -1 )
        self.center_uv_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        right_bsizer2.Add( self.center_uv_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_center_uv.SetSizer( right_bsizer2 )
        self.m_panel_center_uv.Layout()
        right_bsizer2.Fit( self.m_panel_center_uv )
        bottom_row_bSizer.Add( self.m_panel_center_uv, 1, wx.EXPAND, 5 )

        self.m_panel_right_uv = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        ptz_bsizer2 = wx.BoxSizer( wx.VERTICAL )

        self.ptz_image_title2 = wx.StaticText( self.m_panel_right_uv, wx.ID_ANY, u"Right UV", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.ptz_image_title2.Wrap( -1 )
        self.ptz_image_title2.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        ptz_bsizer2.Add( self.ptz_image_title2, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

        bSizer40 = wx.BoxSizer( wx.VERTICAL )

        self.right_uv_panel = wx.Panel( self.m_panel_right_uv, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer40.Add( self.right_uv_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.right_uv_histogram_panel = wx.Panel( self.m_panel_right_uv, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer40.Add( self.right_uv_histogram_panel, 1, wx.EXPAND |wx.ALL, 5 )


        ptz_bsizer2.Add( bSizer40, 1, wx.EXPAND, 5 )

        self.right_uv_status_text = wx.StaticText( self.m_panel_right_uv, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
        self.right_uv_status_text.Wrap( -1 )
        self.right_uv_status_text.SetFont( wx.Font( 10, 70, 90, 92, False, wx.EmptyString ) )

        ptz_bsizer2.Add( self.right_uv_status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.m_panel_right_uv.SetSizer( ptz_bsizer2 )
        self.m_panel_right_uv.Layout()
        ptz_bsizer2.Fit( self.m_panel_right_uv )
        bottom_row_bSizer.Add( self.m_panel_right_uv, 1, wx.EXPAND, 5 )


        bSizer16.Add( bottom_row_bSizer, 1, wx.EXPAND, 5 )

        detector_frames_bSizer11 = wx.BoxSizer( wx.HORIZONTAL )

        self.sys1_detector_frames = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bsizer2131 = wx.BoxSizer( wx.VERTICAL )

        self.left_sys_detector_frames = wx.StaticText( self.sys1_detector_frames, wx.ID_ANY, u"Detector Frames: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.left_sys_detector_frames.Wrap( -1 )
        self.left_sys_detector_frames.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bsizer2131.Add( self.left_sys_detector_frames, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.sys1_detector_frames.SetSizer( bsizer2131 )
        self.sys1_detector_frames.Layout()
        bsizer2131.Fit( self.sys1_detector_frames )
        detector_frames_bSizer11.Add( self.sys1_detector_frames, 1, 0, 5 )

        self.sys0_detector_frame1 = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bsizer21221 = wx.BoxSizer( wx.VERTICAL )

        self.center_sys_detector_frames = wx.StaticText( self.sys0_detector_frame1, wx.ID_ANY, u"Detector Frames: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.center_sys_detector_frames.Wrap( -1 )
        self.center_sys_detector_frames.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bsizer21221.Add( self.center_sys_detector_frames, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.sys0_detector_frame1.SetSizer( bsizer21221 )
        self.sys0_detector_frame1.Layout()
        bsizer21221.Fit( self.sys0_detector_frame1 )
        detector_frames_bSizer11.Add( self.sys0_detector_frame1, 1, 0, 5 )

        self.sys2_detector_frames = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bsizer2121 = wx.BoxSizer( wx.VERTICAL )

        self.right_sys_detector_frames = wx.StaticText( self.sys2_detector_frames, wx.ID_ANY, u"Detector Frames: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.right_sys_detector_frames.Wrap( -1 )
        self.right_sys_detector_frames.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bsizer2121.Add( self.right_sys_detector_frames, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.sys2_detector_frames.SetSizer( bsizer2121 )
        self.sys2_detector_frames.Layout()
        bsizer2121.Fit( self.sys2_detector_frames )
        detector_frames_bSizer11.Add( self.sys2_detector_frames, 1, 0, 5 )


        bSizer16.Add( detector_frames_bSizer11, 0, wx.EXPAND, 5 )

        disk_space_row_bSizer1 = wx.BoxSizer( wx.HORIZONTAL )

        self.sys1_disk_usage_panel = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bsizer213 = wx.BoxSizer( wx.VERTICAL )

        self.left_sys_space_static_text = wx.StaticText( self.sys1_disk_usage_panel, wx.ID_ANY, u"Disk Space: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.left_sys_space_static_text.Wrap( -1 )
        self.left_sys_space_static_text.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bsizer213.Add( self.left_sys_space_static_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.sys1_disk_usage_panel.SetSizer( bsizer213 )
        self.sys1_disk_usage_panel.Layout()
        bsizer213.Fit( self.sys1_disk_usage_panel )
        disk_space_row_bSizer1.Add( self.sys1_disk_usage_panel, 1, 0, 5 )

        self.sys0_disk_usage_panel = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bsizer2122 = wx.BoxSizer( wx.VERTICAL )

        self.center_sys_space_static_text = wx.StaticText( self.sys0_disk_usage_panel, wx.ID_ANY, u"Disk Space: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.center_sys_space_static_text.Wrap( -1 )
        self.center_sys_space_static_text.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bsizer2122.Add( self.center_sys_space_static_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.sys0_disk_usage_panel.SetSizer( bsizer2122 )
        self.sys0_disk_usage_panel.Layout()
        bsizer2122.Fit( self.sys0_disk_usage_panel )
        disk_space_row_bSizer1.Add( self.sys0_disk_usage_panel, 1, 0, 5 )

        self.sys2_disk_usage_panel = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
        bsizer212 = wx.BoxSizer( wx.VERTICAL )

        self.right_sys_space_static_text = wx.StaticText( self.sys2_disk_usage_panel, wx.ID_ANY, u"Disk Space: ?", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.right_sys_space_static_text.Wrap( -1 )
        self.right_sys_space_static_text.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

        bsizer212.Add( self.right_sys_space_static_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        self.sys2_disk_usage_panel.SetSizer( bsizer212 )
        self.sys2_disk_usage_panel.Layout()
        bsizer212.Fit( self.sys2_disk_usage_panel )
        disk_space_row_bSizer1.Add( self.sys2_disk_usage_panel, 1, 0, 5 )


        bSizer16.Add( disk_space_row_bSizer1, 0, wx.EXPAND, 5 )


        self.images_panel.SetSizer( bSizer16 )
        self.images_panel.Layout()
        bSizer16.Fit( self.images_panel )
        bsizer12.Add( self.images_panel, 1, wx.EXPAND|wx.BOTTOM|wx.RIGHT, 5 )


        main_size.Add( bsizer12, 1, wx.EXPAND, 5 )


        self.SetSizer( main_size )
        self.Layout()
        self.m_menubar1 = wx.MenuBar( 0 )
        self.exit_menu = wx.Menu()
        self.exit_menu_item = wx.MenuItem( self.exit_menu, wx.ID_ANY, u"Exit", wx.EmptyString, wx.ITEM_NORMAL )
        self.exit_menu.AppendItem( self.exit_menu_item )

        self.m_menubar1.Append( self.exit_menu, u"File" )

        self.view_menu = wx.Menu()
        self.m_menuItem3 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Show/Hide Left Subsystem", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem3 )

        self.m_menuItem4 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Show/Hide Center Subsystem", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem4 )

        self.m_menuItem5 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Show/Hide Right Subsystem", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem5 )

        self.m_menuItem6 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Show/Hide RGB", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem6 )

        self.m_menuItem7 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Show/Hide IR", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem7 )

        self.m_menuItem8 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Show/Hide UV", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem8 )

        self.m_menuItem9 = wx.MenuItem( self.view_menu, wx.ID_ANY, u"Toggle Saturated Pixels", wx.EmptyString, wx.ITEM_NORMAL )
        self.view_menu.AppendItem( self.m_menuItem9 )

        self.m_menubar1.Append( self.view_menu, u"View" )

        self.calibration_menu = wx.Menu()
        self.m_menuItem19 = wx.MenuItem( self.calibration_menu, wx.ID_ANY, u"Edit System Configurations", wx.EmptyString, wx.ITEM_NORMAL )
        self.calibration_menu.AppendItem( self.m_menuItem19 )

        self.m_menubar1.Append( self.calibration_menu, u"Configuration" )

        self.m_menu_detection = wx.Menu()
        self.m_menu_start_detectors = wx.MenuItem( self.m_menu_detection, wx.ID_ANY, u"Start Detectors", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_start_detectors )

        self.m_menu_stop_detectors = wx.MenuItem( self.m_menu_detection, wx.ID_ANY, u"Stop Detectors", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_stop_detectors )

        self.m_menu_start_detector_sys0 = wx.MenuItem( self.m_menu_detection, ID_START_DETECTOR_SYS0_CENTER, u"Start Detector Sys0 (Center)", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_start_detector_sys0 )

        self.m_menu_start_detector_sys1 = wx.MenuItem( self.m_menu_detection, ID_START_DETECTOR_SYS1_LEFT, u"Start Detector Sys1 (Left)", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_start_detector_sys1 )

        self.m_menu_start_detector_sys2 = wx.MenuItem( self.m_menu_detection, ID_START_DETECTOR_SYS2_RIGHT, u"Start Detector Sys2 (Right)", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_start_detector_sys2 )

        self.m_menu_stop_detector_sys0 = wx.MenuItem( self.m_menu_detection, ID_STOP_DETECTOR_SYS0_CENTER, u"Stop Detector Sys0 (Center)", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_stop_detector_sys0 )

        self.m_menu_stop_detector_sys1 = wx.MenuItem( self.m_menu_detection, ID_STOP_DETECTOR_SYS1_LEFT, u"Stop Detector Sys1 (Left)", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_stop_detector_sys1 )

        self.m_menu_stop_detector_sys2 = wx.MenuItem( self.m_menu_detection, ID_STOP_DETECTOR_SYS2_RIGHT, u"Stop Detector Sys2 (Right)", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu_detection.AppendItem( self.m_menu_stop_detector_sys2 )

        self.m_menubar1.Append( self.m_menu_detection, u"Detection" )

        self.m_menu81 = wx.Menu()
        self.m_menuItem24 = wx.MenuItem( self.m_menu81, wx.ID_ANY, u"System Control Panel", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu81.AppendItem( self.m_menuItem24 )

        self.m_menubar1.Append( self.m_menu81, u"System-Control" )

        self.m_menu8 = wx.Menu()
        self.m_menuItem281 = wx.MenuItem( self.m_menu8, wx.ID_ANY, u"Create Flight Summary", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu8.AppendItem( self.m_menuItem281 )

        self.m_menu_fin_tune_tracking = wx.MenuItem( self.m_menu8, wx._ID_ANY, u"Fine Tune Tracking", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu8.AppendItem( self.m_menu_fin_tune_tracking )

        self.m_menu_detection_summary = wx.MenuItem( self.m_menu8, wx.ID_ANY, u"Detection Summary", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu8.AppendItem( self.m_menu_detection_summary )

        self.view_queue = wx.MenuItem( self.m_menu8, wx.ID_ANY, u"View Queue", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu8.AppendItem( self.view_queue )

        self.clear_queue = wx.MenuItem( self.m_menu8, wx.ID_ANY, u"Clear Queue", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu8.AppendItem( self.clear_queue )

        self.cancel_running_jobs = wx.MenuItem( self.m_menu8, wx.ID_ANY, u"Cancel Running Jobs", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu8.AppendItem( self.cancel_running_jobs )


        self.m_menubar1.Append( self.m_menu8, u"Post-Flight Processing" )

        self.menu_help = wx.Menu()
        self.m_menuItem131 = wx.MenuItem( self.menu_help, wx.ID_ANY, u"Hot Keys", wx.EmptyString, wx.ITEM_NORMAL )
        self.menu_help.AppendItem( self.m_menuItem131 )

        self.menu_item_about = wx.MenuItem( self.menu_help, wx.ID_ANY, u"About", wx.EmptyString, wx.ITEM_NORMAL )
        self.menu_help.AppendItem( self.menu_item_about )

        self.m_menubar1.Append( self.menu_help, u"Help" )

        self.SetMenuBar( self.m_menubar1 )

        self.m_status_bar = self.CreateStatusBar( 1, wx.ST_SIZEGRIP, wx.ID_ANY )

        self.Centre( wx.BOTH )

        # Connect Events
        self.m_button10.Bind( wx.EVT_BUTTON, self.on_set_camera_parameter )
        self.m_manual_ir_nuc.Bind( wx.EVT_BUTTON, self.on_ir_nuc )
        self.effort_combo_box.Bind( wx.EVT_COMBOBOX, self.on_effort_selection )
        self.m_button4.Bind( wx.EVT_BUTTON, self.on_new_effort_metadata_entry )
        self.m_button5.Bind( wx.EVT_BUTTON, self.on_edit_effort_metadata )
        self.m_button6.Bind( wx.EVT_BUTTON, self.on_delete_effort_metadata )
        self.flight_number_text_ctrl.Bind( wx.EVT_TEXT, self.on_update_flight_number )
        self.observer_text_ctrl.Bind( wx.EVT_TEXT, self.on_update_observer )
        self.m_button9.Bind( wx.EVT_BUTTON, self.on_add_to_event_log )
        self.m_button81.Bind( wx.EVT_BUTTON, self.on_set_collection_mode )
        self.start_collecting_button.Bind( wx.EVT_BUTTON, self.start_collecting )
        self.m_button8.Bind( wx.EVT_BUTTON, self.stop_collecting )
        self.start_detectors_button.Bind( wx.EVT_BUTTON, self.on_start_detectors )
        self.stop_detectors_button.Bind( wx.EVT_BUTTON, self.on_stop_detectors )
        self.nas_disk_space.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.close_button.Bind( wx.EVT_BUTTON, self.on_close_button )
        self.camera_config_combo.Bind( wx.EVT_COMBOBOX, self.on_camera_config_combo )
        self.m_panel_left_rgb.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_rgb )
        self.cueing_left_image_title3.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_rgb )
        self.left_rgb_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_rgb )
        self.left_rgb_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_rgb )
        self.left_rgb_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_rgb )
        self.m_panel_center_rgb.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_rgb )
        self.cueing_right_image_title.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_rgb )
        self.center_rgb_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_rgb )
        self.center_rgb_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_rgb )
        self.center_rgb_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_rgb )
        self.m_panel_right_rgb.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_rgb )
        self.ptz_image_title.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_rgb )
        self.right_rgb_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_rgb )
        self.right_rgb_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_rgb )
        self.right_rgb_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_rgb )
        self.m_panel_left_ir.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_ir )
        self.cueing_left_image_title1.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_ir )
        self.left_ir_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_ir )
        self.left_ir_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_ir )
        self.left_ir_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_ir )
        self.m_panel_center_ir.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_ir )
        self.cueing_right_image_title1.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_ir )
        self.center_ir_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_ir )
        self.center_ir_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_ir )
        self.center_ir_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_ir )
        self.m_panel_right_ir.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_ir )
        self.ptz_image_title1.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_ir )
        self.right_ir_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_ir )
        self.right_ir_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_ir )
        self.right_ir_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_ir )
        self.m_panel_left_uv.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.cueing_left_image_title2.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.left_uv_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.left_uv_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.left_uv_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.m_panel_center_uv.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.cueing_right_image_title2.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.center_uv_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.center_uv_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.center_uv_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.m_panel_right_uv.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.ptz_image_title2.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.right_uv_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.right_uv_histogram_panel.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.right_uv_status_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.left_sys_detector_frames.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.center_sys_detector_frames.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.right_sys_detector_frames.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.left_sys_space_static_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_left_uv )
        self.center_sys_space_static_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_center_uv )
        self.right_sys_space_static_text.Bind( wx.EVT_LEFT_DCLICK, self.on_dclick_right_uv )
        self.Bind( wx.EVT_MENU, self.on_close_button, id = self.exit_menu_item.GetId() )
        self.Bind( wx.EVT_MENU, self.on_show_or_hide_left, id = self.m_menuItem3.GetId() )
        self.Bind( wx.EVT_MENU, self.on_show_or_hide_center, id = self.m_menuItem4.GetId() )
        self.Bind( wx.EVT_MENU, self.on_show_or_hide_right, id = self.m_menuItem5.GetId() )
        self.Bind( wx.EVT_MENU, self.on_show_or_hide_rgb, id = self.m_menuItem6.GetId() )
        self.Bind( wx.EVT_MENU, self.on_show_or_hide_ir, id = self.m_menuItem7.GetId() )
        self.Bind( wx.EVT_MENU, self.on_show_or_hide_uv, id = self.m_menuItem8.GetId() )
        self.Bind( wx.EVT_MENU, self.on_toggle_saturated_pixels, id = self.m_menuItem9.GetId() )
        self.Bind( wx.EVT_MENU, self.on_edit_camera_configuration, id = self.m_menuItem19.GetId() )
        self.Bind( wx.EVT_MENU, self.on_system_startup_frame_raise, id = self.m_menuItem24.GetId() )
        self.Bind( wx.EVT_MENU, self.on_create_flight_summary, id = self.m_menuItem281.GetId() )
        self.Bind( wx.EVT_MENU, self.on_measure_image_to_image_homographies, id = self.m_menu_fin_tune_tracking.GetId() )
        self.Bind( wx.EVT_MENU, self.on_detection_summary, id = self.m_menu_detection_summary.GetId() )
        self.Bind( wx.EVT_MENU, self.on_view_queue, id = self.view_queue.GetId() )
        self.Bind( wx.EVT_MENU, self.on_clear_queue, id = self.clear_queue.GetId() )
        self.Bind( wx.EVT_MENU, self.on_cancel_running_jobs, id = self.cancel_running_jobs.GetId() )
        self.Bind( wx.EVT_MENU, self.on_hot_key_help, id = self.m_menuItem131.GetId() )
        self.Bind( wx.EVT_MENU, self.on_menu_item_about, id = self.menu_item_about.GetId() )
        self.Bind( wx.EVT_MENU, self.on_start_detectors, id = self.m_menu_start_detectors.GetId() )
        self.Bind( wx.EVT_MENU, self.on_stop_detectors, id = self.m_menu_stop_detectors.GetId() )
        self.Bind( wx.EVT_MENU, self.on_start_detector_sys0, id = self.m_menu_start_detector_sys0.GetId() )
        self.Bind( wx.EVT_MENU, self.on_start_detector_sys1, id = self.m_menu_start_detector_sys1.GetId() )
        self.Bind( wx.EVT_MENU, self.on_start_detector_sys2, id = self.m_menu_start_detector_sys2.GetId() )
        self.Bind( wx.EVT_MENU, self.on_stop_detector_sys0, id = self.m_menu_stop_detector_sys0.GetId() )
        self.Bind( wx.EVT_MENU, self.on_stop_detector_sys1, id = self.m_menu_stop_detector_sys1.GetId() )
        self.Bind( wx.EVT_MENU, self.on_stop_detector_sys2, id = self.m_menu_stop_detector_sys2.GetId() )

    def __del__( self ):
        # Disconnect Events
        self.m_button10.Unbind( wx.EVT_BUTTON, None )
        self.m_manual_ir_nuc.Unbind( wx.EVT_BUTTON, None )
        self.effort_combo_box.Unbind( wx.EVT_COMBOBOX, None )
        self.m_button4.Unbind( wx.EVT_BUTTON, None )
        self.m_button5.Unbind( wx.EVT_BUTTON, None )
        self.m_button6.Unbind( wx.EVT_BUTTON, None )
        self.flight_number_text_ctrl.Unbind( wx.EVT_TEXT, None )
        self.observer_text_ctrl.Unbind( wx.EVT_TEXT, None )
        self.m_button9.Unbind( wx.EVT_BUTTON, None )
        self.m_button81.Unbind( wx.EVT_BUTTON, None )
        self.start_collecting_button.Unbind( wx.EVT_BUTTON, None )
        self.m_button8.Unbind( wx.EVT_BUTTON, None )
        self.start_detectors_button.Unbind( wx.EVT_BUTTON, None )
        self.stop_detectors_button.Unbind( wx.EVT_BUTTON, None )
        self.nas_disk_space.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.close_button.Unbind( wx.EVT_BUTTON, None )
        self.camera_config_combo.Unbind( wx.EVT_COMBOBOX, None )
        self.m_panel_left_rgb.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.cueing_left_image_title3.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_rgb_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_rgb_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_rgb_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_center_rgb.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.cueing_right_image_title.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_rgb_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_rgb_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_rgb_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_right_rgb.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.ptz_image_title.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_rgb_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_rgb_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_rgb_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_left_ir.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.cueing_left_image_title1.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_ir_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_ir_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_ir_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_center_ir.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.cueing_right_image_title1.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_ir_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_ir_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_ir_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_right_ir.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.ptz_image_title1.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_ir_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_ir_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_ir_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_left_uv.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.cueing_left_image_title2.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_uv_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_uv_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_uv_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_center_uv.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.cueing_right_image_title2.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_uv_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_uv_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_uv_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.m_panel_right_uv.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.ptz_image_title2.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_uv_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_uv_histogram_panel.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_uv_status_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_sys_detector_frames.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_sys_detector_frames.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_sys_detector_frames.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.left_sys_space_static_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.center_sys_space_static_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.right_sys_space_static_text.Unbind( wx.EVT_LEFT_DCLICK, None )
        self.Unbind( wx.EVT_MENU, id = self.exit_menu_item.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem3.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem4.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem5.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem6.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem7.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem8.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem9.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem19.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem24.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem281.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.m_menuItem131.GetId() )
        self.Unbind( wx.EVT_MENU, id = self.menu_item_about.GetId() )


    # Virtual event handlers, overide them in your derived class
    def on_set_camera_parameter( self, event ):
        event.Skip()

    def on_effort_selection( self, event ):
        event.Skip()

    def on_new_effort_metadata_entry( self, event ):
        event.Skip()

    def on_edit_effort_metadata( self, event ):
        event.Skip()

    def on_delete_effort_metadata( self, event ):
        event.Skip()

    def on_update_flight_number( self, event ):
        event.Skip()

    def on_update_observer( self, event ):
        event.Skip()

    def on_add_to_event_log( self, event ):
        event.Skip()

    def on_set_collection_mode( self, event ):
        event.Skip()

    def start_collecting( self, event ):
        event.Skip()

    def on_start_detectors( self, event ):
        event.Skip()

    def on_stop_detectors( self, event ):
        event.Skip()

    def on_start_detector_sys0( self, event):
        event.Skip()

    def on_start_detector_sys1( self, event):
        event.Skip()

    def on_start_detector_sys2( self, event):
        event.Skip()

    def on_stop_detector_sys0( self, event):
        event.Skip()

    def on_stop_detector_sys1( self, event):
        event.Skip()

    def on_stop_detector_sys2( self, event):
        event.Skip()

    def stop_collecting( self, event ):
        event.Skip()

    def on_dclick_left_uv( self, event ):
        event.Skip()

    def on_close_button( self, event ):
        event.Skip()

    def on_camera_config_combo( self, event ):
        event.Skip()

    def on_dclick_left_rgb( self, event ):
        event.Skip()





    def on_dclick_center_rgb( self, event ):
        event.Skip()





    def on_dclick_right_rgb( self, event ):
        event.Skip()





    def on_dclick_left_ir( self, event ):
        event.Skip()





    def on_dclick_center_ir( self, event ):
        event.Skip()





    def on_dclick_right_ir( self, event ):
        event.Skip()










    def on_dclick_center_uv( self, event ):
        event.Skip()





    def on_dclick_right_uv( self, event ):
        event.Skip()












    def on_show_or_hide_left( self, event ):
        event.Skip()

    def on_show_or_hide_center( self, event ):
        event.Skip()

    def on_show_or_hide_right( self, event ):
        event.Skip()

    def on_show_or_hide_rgb( self, event ):
        event.Skip()

    def on_show_or_hide_ir( self, event ):
        event.Skip()

    def on_show_or_hide_uv( self, event ):
        event.Skip()

    def on_toggle_saturated_pixels( self, event ):
        event.Skip()

    def on_edit_camera_configuration( self, event ):
        event.Skip()

    def on_system_startup_frame_raise( self, event ):
        event.Skip()

    def on_create_flight_summary( self, event ):
        event.Skip()

    def on_measure_image_to_image_homographies( self, event ):
        event.Skip()

    def on_detection_summary( self, event ):
        event.Skip()

    def on_view_queue( self, event ):
        event.Skip()

    def on_clear_queue( self, event ):
        event.Skip()

    def on_cancel_running_jobs( self, event ):
        event.Skip()

    def on_hot_key_help( self, event ):
        event.Skip()

    def on_menu_item_about( self, event ):
        event.Skip()

