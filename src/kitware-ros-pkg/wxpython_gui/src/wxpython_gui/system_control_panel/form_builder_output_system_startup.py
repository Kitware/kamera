# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jan 28 2021)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class MainFrame
###########################################################################

class MainFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"System Startup and Control Commands", pos = wx.DefaultPosition, size = wx.Size( 893,701 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.Size( -1,-1 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
		
		main_size = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )
		
		bSizer8 = wx.BoxSizer( wx.VERTICAL )
		
		gSizer1 = wx.GridSizer( 4, 3, 0, 0 )
		
		
		gSizer1.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		self.m_staticText341 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Entire System", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText341.Wrap( -1 )
		self.m_staticText341.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		gSizer1.Add( self.m_staticText341, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.AddSpacer( ( 0, 0), 1, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_button14211 = wx.Button( self.m_panel33, wx.ID_ANY, u"Start Entire System", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer1.Add( self.m_button14211, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_button142111 = wx.Button( self.m_panel33, wx.ID_ANY, u"Stop Entire System", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer1.Add( self.m_button142111, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer1021 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1421 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart Entire System", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1021.Add( self.m_button1421, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer1021, 1, wx.EXPAND, 5 )
		
		bSizer103 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button143 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All Cameras", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer103.Add( self.m_button143, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer103, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer1031 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1431 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All RGB Cameras", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		bSizer1031.Add( self.m_button1431, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer1031, 1, wx.EXPAND, 5 )
		
		bSizer1032 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1432 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All IR Cameras", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1032.Add( self.m_button1432, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer1032, 1, wx.EXPAND, 5 )
		
		bSizer1033 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1433 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All UV Cameras", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1033.Add( self.m_button1433, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer1033, 1, wx.EXPAND, 5 )
		
		bSizer10331 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14331 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart DAQ", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10331.Add( self.m_button14331, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer10331, 1, wx.EXPAND, 5 )
		
		bSizer10332 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14332 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart INS", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10332.Add( self.m_button14332, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer1.Add( bSizer10332, 1, wx.EXPAND, 5 )
		
		
		bSizer8.Add( gSizer1, 0, wx.EXPAND, 5 )
		
		self.m_staticline3 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer8.Add( self.m_staticline3, 0, wx.EXPAND |wx.ALL, 5 )
		
		gSizer11 = wx.GridSizer( 4, 3, 0, 0 )
		
		
		gSizer11.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		self.m_staticText3411 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Left-View Computer (sys1)", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3411.Wrap( -1 )
		self.m_staticText3411.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		gSizer11.Add( self.m_staticText3411, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		bSizer101 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button141 = wx.Button( self.m_panel33, wx.ID_ANY, u"Start All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer101.Add( self.m_button141, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer101, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer1013 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1413 = wx.Button( self.m_panel33, wx.ID_ANY, u"Stop All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1013.Add( self.m_button1413, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer1013, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer1014 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014.Add( self.m_button1414, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer1014, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer10141 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14141 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart RGB Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10141.Add( self.m_button14141, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer10141, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer101411 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button141411 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart IR Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer101411.Add( self.m_button141411, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer101411, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer101412 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button141412 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart UV Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer101412.Add( self.m_button141412, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer101412, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer101414 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button141414 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart Nexus", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer101414.Add( self.m_button141414, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer11.Add( bSizer101414, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer8.Add( gSizer11, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticline2 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer8.Add( self.m_staticline2, 0, wx.EXPAND |wx.ALL, 5 )
		
		gSizer111 = wx.GridSizer( 4, 3, 0, 0 )
		
		
		gSizer111.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		self.m_staticText34111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Center-View Computer (sys0)", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34111.Wrap( -1 )
		self.m_staticText34111.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		gSizer111.Add( self.m_staticText34111, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		bSizer1011 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1411 = wx.Button( self.m_panel33, wx.ID_ANY, u"Start All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1011.Add( self.m_button1411, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer1011, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer10131 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14131 = wx.Button( self.m_panel33, wx.ID_ANY, u"Stop All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10131.Add( self.m_button14131, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer10131, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer10142 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14142 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10142.Add( self.m_button14142, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer10142, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer101413 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button141413 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart RGB Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer101413.Add( self.m_button141413, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer101413, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer1014111 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414111 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart IR Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014111.Add( self.m_button1414111, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer1014111, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer1014121 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414121 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart UV Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014121.Add( self.m_button1414121, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer1014121, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer1014141 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414141 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart Nexus", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014141.Add( self.m_button1414141, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer111.Add( bSizer1014141, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer8.Add( gSizer111, 0, wx.EXPAND, 5 )
		
		self.m_staticline1 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer8.Add( self.m_staticline1, 0, wx.EXPAND |wx.ALL, 5 )
		
		gSizer112 = wx.GridSizer( 4, 3, 0, 0 )
		
		
		gSizer112.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		self.m_staticText34112 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Right-View Computer (sys2)", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34112.Wrap( -1 )
		self.m_staticText34112.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		gSizer112.Add( self.m_staticText34112, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		bSizer1012 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1412 = wx.Button( self.m_panel33, wx.ID_ANY, u"Start All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1012.Add( self.m_button1412, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer1012, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer10132 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14132 = wx.Button( self.m_panel33, wx.ID_ANY, u"Stop All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10132.Add( self.m_button14132, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer10132, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer10143 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button14143 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart All Nodes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10143.Add( self.m_button14143, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer10143, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer101415 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button141415 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart RGB Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer101415.Add( self.m_button141415, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer101415, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer1014112 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414112 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart IR Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014112.Add( self.m_button1414112, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer1014112, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		bSizer1014122 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414122 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart UV Camera", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014122.Add( self.m_button1414122, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer1014122, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer1014142 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1414142 = wx.Button( self.m_panel33, wx.ID_ANY, u"Restart Nexus", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1014142.Add( self.m_button1414142, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		gSizer112.Add( bSizer1014142, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer8.Add( gSizer112, 0, wx.EXPAND, 5 )
		
		
		bSizer44.Add( bSizer8, 1, wx.EXPAND, 5 )
		
		
		self.m_panel33.SetSizer( bSizer44 )
		self.m_panel33.Layout()
		bSizer44.Fit( self.m_panel33 )
		main_size.Add( self.m_panel33, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.SetSizer( main_size )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.m_button14211.Bind( wx.EVT_BUTTON, self.on_start_all_nodes )
		self.m_button142111.Bind( wx.EVT_BUTTON, self.on_stop_all_nodes )
		self.m_button1421.Bind( wx.EVT_BUTTON, self.on_restart_all_nodes )
		self.m_button143.Bind( wx.EVT_BUTTON, self.on_restart_all_cameras )
		self.m_button1431.Bind( wx.EVT_BUTTON, self.on_restart_all_rgb_cameras )
		self.m_button1432.Bind( wx.EVT_BUTTON, self.on_restart_all_ir_cameras )
		self.m_button1433.Bind( wx.EVT_BUTTON, self.on_restart_all_uv_cameras )
		self.m_button14331.Bind( wx.EVT_BUTTON, self.on_restart_daq )
		self.m_button14332.Bind( wx.EVT_BUTTON, self.on_restart_ins )
		self.m_button141.Bind( wx.EVT_BUTTON, self.on_start_all_nodes_sys1 )
		self.m_button1413.Bind( wx.EVT_BUTTON, self.on_stop_all_nodes_sys1 )
		self.m_button1414.Bind( wx.EVT_BUTTON, self.on_restart_all_nodes_sys1 )
		self.m_button14141.Bind( wx.EVT_BUTTON, self.on_restart_rgb_camera_sys1 )
		self.m_button141411.Bind( wx.EVT_BUTTON, self.on_restart_ir_camera_sys1 )
		self.m_button141412.Bind( wx.EVT_BUTTON, self.on_restart_uv_camera_sys1 )
		self.m_button141414.Bind( wx.EVT_BUTTON, self.on_restart_nexus_sys1 )
		self.m_button1411.Bind( wx.EVT_BUTTON, self.on_start_all_nodes_sys0 )
		self.m_button14131.Bind( wx.EVT_BUTTON, self.on_stop_all_nodes_sys0 )
		self.m_button14142.Bind( wx.EVT_BUTTON, self.on_restart_all_nodes_sys0 )
		self.m_button141413.Bind( wx.EVT_BUTTON, self.on_restart_rgb_camera_sys0 )
		self.m_button1414111.Bind( wx.EVT_BUTTON, self.on_restart_ir_camera_sys0 )
		self.m_button1414121.Bind( wx.EVT_BUTTON, self.on_restart_uv_camera_sys0 )
		self.m_button1414141.Bind( wx.EVT_BUTTON, self.on_restart_nexus_sys0 )
		self.m_button1412.Bind( wx.EVT_BUTTON, self.on_start_all_nodes_sys2 )
		self.m_button14132.Bind( wx.EVT_BUTTON, self.on_stop_all_nodes_sys2 )
		self.m_button14143.Bind( wx.EVT_BUTTON, self.on_restart_all_nodes_sys2 )
		self.m_button141415.Bind( wx.EVT_BUTTON, self.on_restart_rgb_camera_sys2 )
		self.m_button1414112.Bind( wx.EVT_BUTTON, self.on_restart_ir_camera_sys2 )
		self.m_button1414122.Bind( wx.EVT_BUTTON, self.on_restart_uv_camera_sys2 )
		self.m_button1414142.Bind( wx.EVT_BUTTON, self.on_restart_nexus_sys2 )
	
	def __del__( self ):
		# Disconnect Events
		self.m_button14211.Unbind( wx.EVT_BUTTON, None )
		self.m_button142111.Unbind( wx.EVT_BUTTON, None )
		self.m_button1421.Unbind( wx.EVT_BUTTON, None )
		self.m_button143.Unbind( wx.EVT_BUTTON, None )
		self.m_button1431.Unbind( wx.EVT_BUTTON, None )
		self.m_button1432.Unbind( wx.EVT_BUTTON, None )
		self.m_button1433.Unbind( wx.EVT_BUTTON, None )
		self.m_button14331.Unbind( wx.EVT_BUTTON, None )
		self.m_button14332.Unbind( wx.EVT_BUTTON, None )
		self.m_button141.Unbind( wx.EVT_BUTTON, None )
		self.m_button1413.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414.Unbind( wx.EVT_BUTTON, None )
		self.m_button14141.Unbind( wx.EVT_BUTTON, None )
		self.m_button141411.Unbind( wx.EVT_BUTTON, None )
		self.m_button141412.Unbind( wx.EVT_BUTTON, None )
		self.m_button141414.Unbind( wx.EVT_BUTTON, None )
		self.m_button1411.Unbind( wx.EVT_BUTTON, None )
		self.m_button14131.Unbind( wx.EVT_BUTTON, None )
		self.m_button14142.Unbind( wx.EVT_BUTTON, None )
		self.m_button141413.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414111.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414121.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414141.Unbind( wx.EVT_BUTTON, None )
		self.m_button1412.Unbind( wx.EVT_BUTTON, None )
		self.m_button14132.Unbind( wx.EVT_BUTTON, None )
		self.m_button14143.Unbind( wx.EVT_BUTTON, None )
		self.m_button141415.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414112.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414122.Unbind( wx.EVT_BUTTON, None )
		self.m_button1414142.Unbind( wx.EVT_BUTTON, None )
	
	
	# Virtual event handlers, overide them in your derived class
	def on_start_all_nodes( self, event ):
		event.Skip()
	
	def on_stop_all_nodes( self, event ):
		event.Skip()
	
	def on_restart_all_nodes( self, event ):
		event.Skip()
	
	def on_restart_all_cameras( self, event ):
		event.Skip()
	
	def on_restart_all_rgb_cameras( self, event ):
		event.Skip()
	
	def on_restart_all_ir_cameras( self, event ):
		event.Skip()
	
	def on_restart_all_uv_cameras( self, event ):
		event.Skip()
	
	def on_restart_daq( self, event ):
		event.Skip()
	
	def on_restart_ins( self, event ):
		event.Skip()
	
	def on_start_all_nodes_sys1( self, event ):
		event.Skip()
	
	def on_stop_all_nodes_sys1( self, event ):
		event.Skip()
	
	def on_restart_all_nodes_sys1( self, event ):
		event.Skip()
	
	def on_restart_rgb_camera_sys1( self, event ):
		event.Skip()
	
	def on_restart_ir_camera_sys1( self, event ):
		event.Skip()
	
	def on_restart_uv_camera_sys1( self, event ):
		event.Skip()
	
	def on_restart_nexus_sys1( self, event ):
		event.Skip()
	
	def on_start_all_nodes_sys0( self, event ):
		event.Skip()
	
	def on_stop_all_nodes_sys0( self, event ):
		event.Skip()
	
	def on_restart_all_nodes_sys0( self, event ):
		event.Skip()
	
	def on_restart_rgb_camera_sys0( self, event ):
		event.Skip()
	
	def on_restart_ir_camera_sys0( self, event ):
		event.Skip()
	
	def on_restart_uv_camera_sys0( self, event ):
		event.Skip()
	
	def on_restart_nexus_sys0( self, event ):
		event.Skip()
	
	def on_start_all_nodes_sys2( self, event ):
		event.Skip()
	
	def on_stop_all_nodes_sys2( self, event ):
		event.Skip()
	
	def on_restart_all_nodes_sys2( self, event ):
		event.Skip()
	
	def on_restart_rgb_camera_sys2( self, event ):
		event.Skip()
	
	def on_restart_ir_camera_sys2( self, event ):
		event.Skip()
	
	def on_restart_uv_camera_sys2( self, event ):
		event.Skip()
	
	def on_restart_nexus_sys2( self, event ):
		event.Skip()
	

