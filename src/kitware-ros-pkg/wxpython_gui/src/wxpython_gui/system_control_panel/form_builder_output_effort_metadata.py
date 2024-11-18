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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Effort Metadata Entry", pos = wx.DefaultPosition, size = wx.Size( 753,473 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHintsSz( wx.Size( 400,400 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )

		main_size = wx.BoxSizer( wx.HORIZONTAL )

		self.m_panel33 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer44 = wx.BoxSizer( wx.VERTICAL )

		self.m_staticText34 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Collection Effort Metadata", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34.Wrap( -1 )
		self.m_staticText34.SetFont( wx.Font( 18, 70, 90, 92, False, wx.EmptyString ) )

		bSizer44.Add( self.m_staticText34, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

		bSizer10 = wx.BoxSizer( wx.HORIZONTAL )

		self.on_populate_from_last_entry_button = wx.Button( self.m_panel33, wx.ID_ANY, u"Autofill from Previous", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer10.Add( self.on_populate_from_last_entry_button, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


		bSizer44.Add( bSizer10, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

		bSizer45 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_staticText33 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Effort Name", wx.DefaultPosition, wx.Size( 180,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText33.Wrap( -1 )
		self.m_staticText33.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer45.Add( self.m_staticText33, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.effort_nickname_textCtrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.effort_nickname_textCtrl.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer45.Add( self.effort_nickname_textCtrl, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		bSizer44.Add( bSizer45, 0, wx.EXPAND, 5 )

		bSizer451 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_staticText331 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Project Name", wx.DefaultPosition, wx.Size( 180,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText331.Wrap( -1 )
		self.m_staticText331.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer451.Add( self.m_staticText331, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.project_name_textCtrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.project_name_textCtrl.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer451.Add( self.project_name_textCtrl, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		bSizer44.Add( bSizer451, 0, wx.EXPAND, 5 )

		bSizer4521 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_staticText3321 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Aircraft", wx.DefaultPosition, wx.Size( 180,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3321.Wrap( -1 )
		self.m_staticText3321.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer4521.Add( self.m_staticText3321, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.aircraft_textCtrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.aircraft_textCtrl.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer4521.Add( self.aircraft_textCtrl, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		bSizer44.Add( bSizer4521, 0, wx.EXPAND, 5 )

		bSizer454 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_staticText334 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Effort\nDescription", wx.DefaultPosition, wx.Size( 180,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText334.Wrap( -1 )
		self.m_staticText334.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer454.Add( self.m_staticText334, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.field_notes_textCtrl = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE )
		self.field_notes_textCtrl.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer454.Add( self.field_notes_textCtrl, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )


		bSizer44.Add( bSizer454, 1, wx.EXPAND, 5 )

		bSizer4541 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_staticText3341 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Image-Process\nWait Time (s)", wx.DefaultPosition, wx.Size( 180,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText3341.Wrap( -1 )
		self.m_staticText3341.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer4541.Add( self.m_staticText3341, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.wait_time_sec = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.wait_time_sec.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer4541.Add( self.wait_time_sec, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.m_staticText33411 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Delete Old\nImages (s)", wx.DefaultPosition, wx.Size( 120,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText33411.Wrap( -1 )
		self.m_staticText33411.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer4541.Add( self.m_staticText33411, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.delete_old_images_sec = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.delete_old_images_sec.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer4541.Add( self.delete_old_images_sec, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.m_staticText33412 = wx.StaticText( self.m_panel33, wx.ID_ANY, u"Save every\nNth Image", wx.DefaultPosition, wx.Size( 140,-1 ), wx.ALIGN_CENTRE )
		self.m_staticText33412.Wrap( -1 )
		self.m_staticText33412.SetFont( wx.Font( 14, 70, 90, 92, False, wx.EmptyString ) )

		bSizer4541.Add( self.m_staticText33412, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.save_every_x_image = wx.TextCtrl( self.m_panel33, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.save_every_x_image.SetFont( wx.Font( 14, 70, 90, 90, False, wx.EmptyString ) )

		bSizer4541.Add( self.save_every_x_image, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		bSizer44.Add( bSizer4541, 1, wx.EXPAND, 5 )

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
		self.on_populate_from_last_entry_button.Bind( wx.EVT_BUTTON, self.on_populate_from_last_entry )
		self.m_button6.Bind( wx.EVT_BUTTON, self.on_save )
		self.m_button7.Bind( wx.EVT_BUTTON, self.on_cancel )

	def __del__( self ):
		# Disconnect Events
		self.on_populate_from_last_entry_button.Unbind( wx.EVT_BUTTON, None )
		self.m_button6.Unbind( wx.EVT_BUTTON, None )
		self.m_button7.Unbind( wx.EVT_BUTTON, None )


	# Virtual event handlers, overide them in your derived class
	def on_populate_from_last_entry( self, event ):
		event.Skip()

	def on_save( self, event ):
		event.Skip()

	def on_cancel( self, event ):
		event.Skip()


