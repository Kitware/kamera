# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Feb 16 2016)
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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Event Log Note", pos = wx.DefaultPosition, size = wx.Size( 700,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.Size( 400,300 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
		
		main_size = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText34 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Add Note to Event Log", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34.Wrap( -1 )
		self.m_staticText34.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer44.Add( self.m_staticText34, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText8 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Note is automatically timestamped", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )
		bSizer44.Add( self.m_staticText8, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer454 = wx.BoxSizer( wx.VERTICAL )
		
		bSizer5 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText3 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Event Type:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )
		self.m_staticText3.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer5.Add( self.m_staticText3, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		event_type_comboChoices = [ u"Takeoff airport", u"Camera failure (position, type)", u"Camera recovered (position, type)", u"Landing airport", u"Other" ]
		self.event_type_combo = wx.ComboBox( self.m_panel33, wx.ID_ANY, u"Other", wx.DefaultPosition, wx.Size( 400,-1 ), event_type_comboChoices, wx.CB_READONLY )
		self.event_type_combo.SetSelection( 4 )
		bSizer5.Add( self.event_type_combo, 0, wx.ALL, 5 )
		
		
		bSizer454.Add( bSizer5, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.note_textCtrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE )
		self.note_textCtrl.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )
		
		bSizer454.Add( self.note_textCtrl, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )
		
		
		bSizer44.Add( bSizer454, 1, wx.EXPAND, 5 )
		
		bSizer62 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_button6 = wx.Button( self.m_panel33, wx.ID_ANY, u"Save", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer62.Add( self.m_button6, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
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
		self.m_button6.Bind( wx.EVT_BUTTON, self.on_save )
		self.m_button7.Bind( wx.EVT_BUTTON, self.on_cancel )
	
	def __del__( self ):
		# Disconnect Events
		self.m_button6.Unbind( wx.EVT_BUTTON, None )
		self.m_button7.Unbind( wx.EVT_BUTTON, None )
	
	
	# Virtual event handlers, overide them in your derived class
	def on_save( self, event ):
		event.Skip()
	
	def on_cancel( self, event ):
		event.Skip()
	

