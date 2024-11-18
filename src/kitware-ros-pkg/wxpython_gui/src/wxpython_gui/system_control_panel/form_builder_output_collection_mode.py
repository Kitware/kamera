# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jan 30 2023)
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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Collection Mode", pos = wx.DefaultPosition, size = wx.Size( 517,350 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.Size( -1,-1 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
		
		main_size = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText34 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Set Collection Mode", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34.Wrap( -1 )
		self.m_staticText34.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer44.Add( self.m_staticText34, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer454 = wx.BoxSizer( wx.VERTICAL )
		
		bSizer5 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Collection Mode:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )
		self.m_staticText3.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer5.Add( self.m_staticText3, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		mode_combo_boxChoices = [ u"fixed rate", u"fixed overlap" ]
		self.mode_combo_box = wx.ComboBox( self.m_panel33, wx.ID_ANY, u"fixed image overlap", wx.DefaultPosition, wx.Size( 250,-1 ), mode_combo_boxChoices, 0 )
		self.mode_combo_box.SetSelection( 1 )
		bSizer5.Add( self.mode_combo_box, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer454.Add( bSizer5, 1, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.percent_panel = wx.Panel( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer61 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText41 = wx.StaticText( self.percent_panel, wx.ID_ANY, u"Overlap (percent)", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText41.Wrap( -1 )
		self.m_staticText41.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer61.Add( self.m_staticText41, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.overlap_txtctrl = wx.TextCtrl( self.percent_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer61.Add( self.overlap_txtctrl, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		self.percent_panel.SetSizer( bSizer61 )
		self.percent_panel.Layout()
		bSizer61.Fit( self.percent_panel )
		bSizer454.Add( self.percent_panel, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.rate_panel = wx.Panel( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer6 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText4 = wx.StaticText( self.rate_panel, wx.ID_ANY, u"Rate (fps)", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )
		self.m_staticText4.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer6.Add( self.m_staticText4, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.rate_txtctrl = wx.TextCtrl( self.rate_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer6.Add( self.rate_txtctrl, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.allow_nuc = wx.CheckBox( self.rate_panel, wx.ID_ANY, u"Allow NUC", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer6.Add( self.allow_nuc, 0, wx.ALL, 5 )
		
		
		self.rate_panel.SetSizer( bSizer6 )
		self.rate_panel.Layout()
		bSizer6.Fit( self.rate_panel )
		bSizer454.Add( self.rate_panel, 1, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticline2 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer454.Add( self.m_staticline2, 0, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel4 = wx.Panel( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer8 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText5 = wx.StaticText( self.m_panel4, wx.ID_ANY, u"Select a shapefile to define the regions to collect imagery", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText5.Wrap( -1 )
		bSizer8.Add( self.m_staticText5, 0, wx.ALL, 5 )
		
		bSizer9 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.shapefile_file_picker = wx.FilePickerCtrl( self.m_panel4, wx.ID_ANY, u"/home/user/kamera_ws/src/kitware-ros-pkg/wxpython_gui/shapefiles/Kotz_TestPoly.shp", u"Select a file", u"*.*", wx.DefaultPosition, wx.Size( 300,-1 ), wx.FLP_DEFAULT_STYLE )
		bSizer9.Add( self.shapefile_file_picker, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.shapefile_checkbox = wx.CheckBox( self.m_panel4, wx.ID_ANY, u"Activate", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer9.Add( self.shapefile_checkbox, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer8.Add( bSizer9, 1, wx.EXPAND, 5 )
		
		
		self.m_panel4.SetSizer( bSizer8 )
		self.m_panel4.Layout()
		bSizer8.Fit( self.m_panel4 )
		bSizer454.Add( self.m_panel4, 1, wx.ALL|wx.EXPAND, 5 )
		
		self.m_staticline1 = wx.StaticLine( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer454.Add( self.m_staticline1, 0, wx.EXPAND |wx.ALL, 5 )
		
		
		bSizer44.Add( bSizer454, 1, wx.EXPAND, 5 )
		
		bSizer62 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.save_button = wx.Button( self.m_panel33, wx.ID_ANY, u"Save", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer62.Add( self.save_button, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.m_button7 = wx.Button( self.m_panel33, wx.ID_ANY, u"Cancel", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer62.Add( self.m_button7, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		
		bSizer44.Add( bSizer62, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel33.SetSizer( bSizer44 )
		self.m_panel33.Layout()
		bSizer44.Fit( self.m_panel33 )
		main_size.Add( self.m_panel33, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.SetSizer( main_size )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.mode_combo_box.Bind( wx.EVT_COMBOBOX, self.on_set_mode )
		self.shapefile_file_picker.Bind( wx.EVT_FILEPICKER_CHANGED, self.on_select_shapefile )
		self.save_button.Bind( wx.EVT_BUTTON, self.on_save )
		self.m_button7.Bind( wx.EVT_BUTTON, self.on_cancel )
	
	def __del__( self ):
		# Disconnect Events
		self.mode_combo_box.Unbind( wx.EVT_COMBOBOX, None )
		self.shapefile_file_picker.Unbind( wx.EVT_FILEPICKER_CHANGED, None )
		self.save_button.Unbind( wx.EVT_BUTTON, None )
		self.m_button7.Unbind( wx.EVT_BUTTON, None )
	
	
	# Virtual event handlers, overide them in your derived class
	def on_set_mode( self, event ):
		event.Skip()
	
	def on_select_shapefile( self, event ):
		event.Skip()
	
	def on_save( self, event ):
		event.Skip()
	
	def on_cancel( self, event ):
		event.Skip()
	

