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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"System Configuration Editor", pos = wx.DefaultPosition, size = wx.Size( 994,969 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.Size( 400,400 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
		
		main_size = wx.BoxSizer( wx.VERTICAL )
		
		self.m_panel2 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer22 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText34 = wx.StaticText( self.m_panel2, wx.ID_ANY, u"System Configuration", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34.Wrap( -1 )
		self.m_staticText34.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer22.Add( self.m_staticText34, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer9 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText6 = wx.StaticText( self.m_panel2, wx.ID_ANY, u"Select\nConfiguration", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
		self.m_staticText6.Wrap( -1 )
		self.m_staticText6.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer9.Add( self.m_staticText6, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		camera_config_comboChoices = []
		self.camera_config_combo = wx.ComboBox( self.m_panel2, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 400,-1 ), camera_config_comboChoices, 0 )
		bSizer9.Add( self.camera_config_combo, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer9.AddSpacer( ( 20, 0), 1, wx.EXPAND, 5 )
		
		self.m_button4 = wx.Button( self.m_panel2, wx.ID_ANY, u"New", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer9.Add( self.m_button4, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer9.AddSpacer( ( 20, 0), 1, wx.EXPAND, 5 )
		
		self.m_button41 = wx.Button( self.m_panel2, wx.ID_ANY, u"New from Current", wx.DefaultPosition, wx.Size( 150,-1 ), 0 )
		bSizer9.Add( self.m_button41, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer9.AddSpacer( ( 20, 0), 1, wx.EXPAND, 5 )
		
		self.m_button42 = wx.Button( self.m_panel2, wx.ID_ANY, u"Delete", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer9.Add( self.m_button42, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer22.Add( bSizer9, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer23 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_button7 = wx.Button( self.m_panel2, wx.ID_ANY, u"Finished Making Changes", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer23.Add( self.m_button7, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText20 = wx.StaticText( self.m_panel2, wx.ID_ANY, u"Make sure to save changes first", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText20.Wrap( -1 )
		bSizer23.Add( self.m_staticText20, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer22.Add( bSizer23, 1, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel2.SetSizer( bSizer22 )
		self.m_panel2.Layout()
		bSizer22.Fit( self.m_panel2 )
		main_size.Add( self.m_panel2, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )
		
		bSizer102 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText3412 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"System Configuration Name", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3412.Wrap( -1 )
		self.m_staticText3412.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer102.Add( self.m_staticText3412, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.config_name_txt_ctrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 500,-1 ), 0 )
		bSizer102.Add( self.config_name_txt_ctrl, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer44.Add( bSizer102, 1, wx.EXPAND, 5 )
		
		self.m_staticline1 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer44.Add( self.m_staticline1, 0, wx.EXPAND |wx.ALL, 5 )
		
		bSizer10 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText341 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Left-View System", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText341.Wrap( -1 )
		self.m_staticText341.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer10.Add( self.m_staticText341, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer44.Add( bSizer10, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer45 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText33 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Left RGB .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText33.Wrap( -1 )
		self.m_staticText33.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer45.Add( self.m_staticText33, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.left_rgb_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45.Add( self.left_rgb_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button18 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45.Add( self.m_button18, 0, wx.ALL, 5 )
		
		self.m_button61 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45.Add( self.m_button61, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer45, 0, wx.EXPAND, 5 )
		
		bSizer452 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText332 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Left IR .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText332.Wrap( -1 )
		self.m_staticText332.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer452.Add( self.m_staticText332, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.left_ir_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer452.Add( self.left_ir_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button19 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer452.Add( self.m_button19, 0, wx.ALL, 5 )
		
		self.m_button611 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer452.Add( self.m_button611, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer452, 0, wx.EXPAND, 5 )
		
		bSizer451 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText331 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Left UV .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText331.Wrap( -1 )
		self.m_staticText331.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer451.Add( self.m_staticText331, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.left_uv_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer451.Add( self.left_uv_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button20 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer451.Add( self.m_button20, 0, wx.ALL, 5 )
		
		self.m_button612 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer451.Add( self.m_button612, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer451, 0, wx.EXPAND, 5 )
		
		bSizer4511 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3311 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Detection .pipe", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3311.Wrap( -1 )
		self.m_staticText3311.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4511.Add( self.m_staticText3311, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.left_pipe_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4511.Add( self.left_pipe_picker, 1, wx.ALL, 5 )
		
		self.m_button21 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4511.Add( self.m_button21, 0, wx.ALL, 5 )
		
		self.m_button613 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4511.Add( self.m_button613, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4511, 0, wx.EXPAND, 5 )
		
		self.m_staticline11 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer44.Add( self.m_staticline11, 0, wx.EXPAND |wx.ALL, 5 )
		
		bSizer101 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText3411 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Center-View System", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3411.Wrap( -1 )
		self.m_staticText3411.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer101.Add( self.m_staticText3411, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer44.Add( bSizer101, 0, wx.EXPAND, 5 )
		
		bSizer4521 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3321 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Center RGB.yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3321.Wrap( -1 )
		self.m_staticText3321.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4521.Add( self.m_staticText3321, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.center_rgb_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4521.Add( self.center_rgb_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button29 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4521.Add( self.m_button29, 0, wx.ALL, 5 )
		
		self.m_button614 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4521.Add( self.m_button614, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4521, 0, wx.EXPAND, 5 )
		
		bSizer4522 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3322 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Center IR .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3322.Wrap( -1 )
		self.m_staticText3322.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4522.Add( self.m_staticText3322, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.center_ir_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4522.Add( self.center_ir_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button27 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4522.Add( self.m_button27, 0, wx.ALL, 5 )
		
		self.m_button615 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4522.Add( self.m_button615, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4522, 0, wx.EXPAND, 5 )
		
		bSizer4523 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3323 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Center UV .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3323.Wrap( -1 )
		self.m_staticText3323.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4523.Add( self.m_staticText3323, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.center_uv_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4523.Add( self.center_uv_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button28 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4523.Add( self.m_button28, 0, wx.ALL, 5 )
		
		self.m_button616 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4523.Add( self.m_button616, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4523, 0, wx.EXPAND, 5 )
		
		bSizer45111 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText33111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Detection .pipe", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText33111.Wrap( -1 )
		self.m_staticText33111.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer45111.Add( self.m_staticText33111, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.center_pipe_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45111.Add( self.center_pipe_picker, 1, wx.ALL, 5 )
		
		self.m_button26 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45111.Add( self.m_button26, 0, wx.ALL, 5 )
		
		self.m_button617 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45111.Add( self.m_button617, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer45111, 0, wx.EXPAND, 5 )
		
		self.m_staticline111 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer44.Add( self.m_staticline111, 0, wx.EXPAND |wx.ALL, 5 )
		
		bSizer1011 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText34111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Right-View System", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34111.Wrap( -1 )
		self.m_staticText34111.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer1011.Add( self.m_staticText34111, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		bSizer44.Add( bSizer1011, 0, wx.EXPAND, 5 )
		
		bSizer4524 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3324 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Right RGB .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3324.Wrap( -1 )
		self.m_staticText3324.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4524.Add( self.m_staticText3324, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.right_rgb_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4524.Add( self.right_rgb_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button25 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4524.Add( self.m_button25, 0, wx.ALL, 5 )
		
		self.m_button618 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4524.Add( self.m_button618, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4524, 0, wx.EXPAND, 5 )
		
		bSizer4525 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3325 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Right IR .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3325.Wrap( -1 )
		self.m_staticText3325.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4525.Add( self.m_staticText3325, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.right_ir_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4525.Add( self.right_ir_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button24 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4525.Add( self.m_button24, 0, wx.ALL, 5 )
		
		self.m_button619 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4525.Add( self.m_button619, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4525, 0, wx.EXPAND, 5 )
		
		bSizer4526 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3326 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Right UV .yaml", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3326.Wrap( -1 )
		self.m_staticText3326.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer4526.Add( self.m_staticText3326, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.right_uv_yaml_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4526.Add( self.right_uv_yaml_picker, 1, wx.ALL, 5 )
		
		self.m_button23 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4526.Add( self.m_button23, 0, wx.ALL, 5 )
		
		self.m_button6110 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer4526.Add( self.m_button6110, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer4526, 0, wx.EXPAND, 5 )
		
		bSizer45112 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText33112 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Detection .pipe", wx.DefaultPosition, wx.Size( 190,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText33112.Wrap( -1 )
		self.m_staticText33112.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer45112.Add( self.m_staticText33112, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.right_pipe_picker = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45112.Add( self.right_pipe_picker, 1, wx.ALL, 5 )
		
		self.m_button22 = wx.Button( self.m_panel33, wx.ID_ANY, u"Find File", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45112.Add( self.m_button22, 0, wx.ALL, 5 )
		
		self.m_button6111 = wx.Button( self.m_panel33, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer45112.Add( self.m_button6111, 0, wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer45112, 0, wx.EXPAND, 5 )
		
		self.m_staticline1111 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer44.Add( self.m_staticline1111, 0, wx.EXPAND |wx.ALL, 5 )
		
		bSizer454 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText334 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Configuration\nDescription", wx.DefaultPosition, wx.Size( -1,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText334.Wrap( -1 )
		self.m_staticText334.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer454.Add( self.m_staticText334, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.configuration_notes_txt_ctrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE )
		self.configuration_notes_txt_ctrl.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )
		
		bSizer454.Add( self.configuration_notes_txt_ctrl, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )
		
		
		bSizer44.Add( bSizer454, 1, wx.EXPAND|wx.LEFT, 5 )
		
		bSizer62 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_button6 = wx.Button( self.m_panel33, wx.ID_ANY, u"Save", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer62.Add( self.m_button6, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer44.Add( bSizer62, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel33.SetSizer( bSizer44 )
		self.m_panel33.Layout()
		bSizer44.Fit( self.m_panel33 )
		main_size.Add( self.m_panel33, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.SetSizer( main_size )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.camera_config_combo.Bind( wx.EVT_COMBOBOX, self.on_combo_select )
		self.m_button4.Bind( wx.EVT_BUTTON, self.on_new )
		self.m_button41.Bind( wx.EVT_BUTTON, self.on_new_from_current )
		self.m_button42.Bind( wx.EVT_BUTTON, self.on_delete )
		self.m_button7.Bind( wx.EVT_BUTTON, self.on_done )
		self.m_button18.Bind( wx.EVT_BUTTON, self.on_find_left_rgb_yaml )
		self.m_button61.Bind( wx.EVT_BUTTON, self.on_clear_left_rgb_yaml )
		self.m_button19.Bind( wx.EVT_BUTTON, self.on_find_left_ir_yaml )
		self.m_button611.Bind( wx.EVT_BUTTON, self.on_clear_left_ir_yaml )
		self.m_button20.Bind( wx.EVT_BUTTON, self.on_find_left_uv_yaml )
		self.m_button612.Bind( wx.EVT_BUTTON, self.on_clear_left_uv_yaml )
		self.m_button21.Bind( wx.EVT_BUTTON, self.on_find_left_pipe )
		self.m_button613.Bind( wx.EVT_BUTTON, self.on_clear_left_pipe )
		self.m_button29.Bind( wx.EVT_BUTTON, self.on_find_center_rgb_yaml )
		self.m_button614.Bind( wx.EVT_BUTTON, self.on_clear_center_rgb_yaml )
		self.m_button27.Bind( wx.EVT_BUTTON, self.on_find_center_ir_yaml )
		self.m_button615.Bind( wx.EVT_BUTTON, self.on_clear_center_ir_yaml )
		self.m_button28.Bind( wx.EVT_BUTTON, self.on_find_center_uv_yaml )
		self.m_button616.Bind( wx.EVT_BUTTON, self.on_clear_center_uv_yaml )
		self.m_button26.Bind( wx.EVT_BUTTON, self.on_find_center_pipe )
		self.m_button617.Bind( wx.EVT_BUTTON, self.on_clear_center_pipe )
		self.m_button25.Bind( wx.EVT_BUTTON, self.on_find_right_rgb_yaml )
		self.m_button618.Bind( wx.EVT_BUTTON, self.on_clear_right_rgb_yaml )
		self.m_button24.Bind( wx.EVT_BUTTON, self.on_find_right_ir_yaml )
		self.m_button619.Bind( wx.EVT_BUTTON, self.on_clear_right_ir_yaml )
		self.m_button23.Bind( wx.EVT_BUTTON, self.on_find_right_uv_yaml )
		self.m_button6110.Bind( wx.EVT_BUTTON, self.on_clear_right_uv_yaml )
		self.m_button22.Bind( wx.EVT_BUTTON, self.on_find_right_pipe )
		self.m_button6111.Bind( wx.EVT_BUTTON, self.on_clear_right_pipe )
		self.m_button6.Bind( wx.EVT_BUTTON, self.on_save )
	
	def __del__( self ):
		# Disconnect Events
		self.camera_config_combo.Unbind( wx.EVT_COMBOBOX, None )
		self.m_button4.Unbind( wx.EVT_BUTTON, None )
		self.m_button41.Unbind( wx.EVT_BUTTON, None )
		self.m_button42.Unbind( wx.EVT_BUTTON, None )
		self.m_button7.Unbind( wx.EVT_BUTTON, None )
		self.m_button18.Unbind( wx.EVT_BUTTON, None )
		self.m_button61.Unbind( wx.EVT_BUTTON, None )
		self.m_button19.Unbind( wx.EVT_BUTTON, None )
		self.m_button611.Unbind( wx.EVT_BUTTON, None )
		self.m_button20.Unbind( wx.EVT_BUTTON, None )
		self.m_button612.Unbind( wx.EVT_BUTTON, None )
		self.m_button21.Unbind( wx.EVT_BUTTON, None )
		self.m_button613.Unbind( wx.EVT_BUTTON, None )
		self.m_button29.Unbind( wx.EVT_BUTTON, None )
		self.m_button614.Unbind( wx.EVT_BUTTON, None )
		self.m_button27.Unbind( wx.EVT_BUTTON, None )
		self.m_button615.Unbind( wx.EVT_BUTTON, None )
		self.m_button28.Unbind( wx.EVT_BUTTON, None )
		self.m_button616.Unbind( wx.EVT_BUTTON, None )
		self.m_button26.Unbind( wx.EVT_BUTTON, None )
		self.m_button617.Unbind( wx.EVT_BUTTON, None )
		self.m_button25.Unbind( wx.EVT_BUTTON, None )
		self.m_button618.Unbind( wx.EVT_BUTTON, None )
		self.m_button24.Unbind( wx.EVT_BUTTON, None )
		self.m_button619.Unbind( wx.EVT_BUTTON, None )
		self.m_button23.Unbind( wx.EVT_BUTTON, None )
		self.m_button6110.Unbind( wx.EVT_BUTTON, None )
		self.m_button22.Unbind( wx.EVT_BUTTON, None )
		self.m_button6111.Unbind( wx.EVT_BUTTON, None )
		self.m_button6.Unbind( wx.EVT_BUTTON, None )
	
	
	# Virtual event handlers, overide them in your derived class
	def on_combo_select( self, event ):
		event.Skip()
	
	def on_new( self, event ):
		event.Skip()
	
	def on_new_from_current( self, event ):
		event.Skip()
	
	def on_delete( self, event ):
		event.Skip()
	
	def on_done( self, event ):
		event.Skip()
	
	def on_find_left_rgb_yaml( self, event ):
		event.Skip()
	
	def on_clear_left_rgb_yaml( self, event ):
		event.Skip()
	
	def on_find_left_ir_yaml( self, event ):
		event.Skip()
	
	def on_clear_left_ir_yaml( self, event ):
		event.Skip()
	
	def on_find_left_uv_yaml( self, event ):
		event.Skip()
	
	def on_clear_left_uv_yaml( self, event ):
		event.Skip()
	
	def on_find_left_pipe( self, event ):
		event.Skip()
	
	def on_clear_left_pipe( self, event ):
		event.Skip()
	
	def on_find_center_rgb_yaml( self, event ):
		event.Skip()
	
	def on_clear_center_rgb_yaml( self, event ):
		event.Skip()
	
	def on_find_center_ir_yaml( self, event ):
		event.Skip()
	
	def on_clear_center_ir_yaml( self, event ):
		event.Skip()
	
	def on_find_center_uv_yaml( self, event ):
		event.Skip()
	
	def on_clear_center_uv_yaml( self, event ):
		event.Skip()
	
	def on_find_center_pipe( self, event ):
		event.Skip()
	
	def on_clear_center_pipe( self, event ):
		event.Skip()
	
	def on_find_right_rgb_yaml( self, event ):
		event.Skip()
	
	def on_clear_right_rgb_yaml( self, event ):
		event.Skip()
	
	def on_find_right_ir_yaml( self, event ):
		event.Skip()
	
	def on_clear_right_ir_yaml( self, event ):
		event.Skip()
	
	def on_find_right_uv_yaml( self, event ):
		event.Skip()
	
	def on_clear_right_uv_yaml( self, event ):
		event.Skip()
	
	def on_find_right_pipe( self, event ):
		event.Skip()
	
	def on_clear_right_pipe( self, event ):
		event.Skip()
	
	def on_save( self, event ):
		event.Skip()
	

