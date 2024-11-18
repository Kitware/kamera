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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Hot Key List", pos = wx.DefaultPosition, size = wx.Size( 400,377 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.Size( 400,450 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
		
		main_size = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText34 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Hot Keys", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34.Wrap( -1 )
		self.m_staticText34.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer44.Add( self.m_staticText34, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer454 = wx.BoxSizer( wx.VERTICAL )
		
		gbSizer1 = wx.GridBagSizer( 0, 0 )
		gbSizer1.SetFlexibleDirection( wx.BOTH )
		gbSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
		self.m_staticText4 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+h", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )
		self.m_staticText4.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText4, wx.GBPosition( 0, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText5 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"This hot key list", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText5.Wrap( -1 )
		gbSizer1.Add( self.m_staticText5, wx.GBPosition( 0, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText41 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+e", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText41.Wrap( -1 )
		self.m_staticText41.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText41, wx.GBPosition( 1, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText5 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Set context to exposure entry", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText5.Wrap( -1 )
		gbSizer1.Add( self.m_staticText5, wx.GBPosition( 1, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText42 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+s", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText42.Wrap( -1 )
		self.m_staticText42.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText42, wx.GBPosition( 2, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText422 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+d", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText422.Wrap( -1 )
		self.m_staticText422.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText422, wx.GBPosition( 3, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText4221 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+f", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4221.Wrap( -1 )
		self.m_staticText4221.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText4221, wx.GBPosition( 4, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText42211 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+o", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText42211.Wrap( -1 )
		self.m_staticText42211.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )

		gbSizer1.Add( self.m_staticText42211, wx.GBPosition( 5, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
                self.m_staticText422111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+p", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText422111.Wrap( -1 )
		self.m_staticText422111.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText422111, wx.GBPosition( 6, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText421111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+i", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText421111.Wrap( -1 )
		self.m_staticText421111.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
                gbSizer1.Add( self.m_staticText421111, wx.GBPosition( 7, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText42111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+k", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText42111.Wrap( -1 )
		self.m_staticText42111.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText42111, wx.GBPosition( 8, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText52 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Start/stop collecting data", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText52.Wrap( -1 )
		gbSizer1.Add( self.m_staticText52, wx.GBPosition( 2, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText521 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Start detectors", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText521.Wrap( -1 )
		gbSizer1.Add( self.m_staticText521, wx.GBPosition( 3, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText5211 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Stop detectors", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText5211.Wrap( -1 )
		gbSizer1.Add( self.m_staticText5211, wx.GBPosition( 4, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText52111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Previous Camera Configuration", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText52111.Wrap( -1 )
		gbSizer1.Add( self.m_staticText52111, wx.GBPosition( 5, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText521111 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Next Camera Configuration", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText521111.Wrap( -1 )
		gbSizer1.Add( self.m_staticText521111, wx.GBPosition( 6, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText52191 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Previous Effort Configuration", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText52191.Wrap( -1 )
		gbSizer1.Add( self.m_staticText52191, wx.GBPosition( 7, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText521011 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Next Effort Configuration", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText521011.Wrap( -1 )
		gbSizer1.Add( self.m_staticText521011, wx.GBPosition( 8, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText421 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"ctrl+n", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
		self.m_staticText421.Wrap( -1 )
		self.m_staticText421.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText421, wx.GBPosition( 9, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText51 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Add note to log", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText51.Wrap( -1 )
		gbSizer1.Add( self.m_staticText51, wx.GBPosition( 9, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		self.m_staticText4211 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"alt+F4", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
		self.m_staticText4211.Wrap( -1 )
		self.m_staticText4211.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 92, False, wx.EmptyString ) )
		
		gbSizer1.Add( self.m_staticText4211, wx.GBPosition( 10, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText511 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Close", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.m_staticText511.Wrap( -1 )
		gbSizer1.Add( self.m_staticText511, wx.GBPosition( 10, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )
		
		
		bSizer454.Add( gbSizer1, 0, wx.EXPAND, 5 )
		
		
		bSizer44.Add( bSizer454, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer62 = wx.BoxSizer( wx.HORIZONTAL )
		
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
		self.m_button7.Bind( wx.EVT_BUTTON, self.on_cancel )
	
	def __del__( self ):
		# Disconnect Events
		self.m_button7.Unbind( wx.EVT_BUTTON, None )
	
	
	# Virtual event handlers, overide them in your derived class
	def on_cancel( self, event ):
		event.Skip()
	

