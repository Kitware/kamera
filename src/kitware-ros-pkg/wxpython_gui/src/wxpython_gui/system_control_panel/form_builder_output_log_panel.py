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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Log Messages", pos = wx.DefaultPosition, size = wx.Size( 1500,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.Size( 400,300 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
		
		main_size = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )
		
		self.message_panel = wx.Panel( self.m_panel33, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer44.Add( self.message_panel, 1, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer62 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_button7 = wx.Button( self.m_panel33, wx.ID_ANY, u"Close", wx.DefaultPosition, wx.DefaultSize, 0 )
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
		self.m_button7.Bind( wx.EVT_BUTTON, self.on_cancel )
	
	def __del__( self ):
		# Disconnect Events
		self.m_button7.Unbind( wx.EVT_BUTTON, None )
	
	
	# Virtual event handlers, overide them in your derived class
	def on_cancel( self, event ):
		event.Skip()
	

