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
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"System Control Panel", pos = wx.DefaultPosition, size = wx.Size( 1538,900 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHintsSz( wx.Size( 400,400 ), wx.DefaultSize )
		self.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), 70, 90, 90, False, wx.EmptyString ) )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )

		main_size = wx.BoxSizer( wx.HORIZONTAL )

		bSizer20 = wx.BoxSizer( wx.VERTICAL )

		self.image_stream_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer191 = wx.BoxSizer( wx.VERTICAL )

		self.m_staticText142 = wx.StaticText( self.image_stream_panel, wx.ID_ANY, u"Image Stream", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText142.Wrap( -1 )
		self.m_staticText142.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

		bSizer191.Add( self.m_staticText142, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5 )

		image_stream_combo_boxChoices = [ u"Left RGB", u"Center RGB", u"Right RGB", u"Left IR", u"Center IR", u"Right IR", u"Left UV", u"Center UV", u"Right UV" ]
		self.image_stream_combo_box = wx.ComboBox( self.image_stream_panel, wx.ID_ANY, u"Left RGB", wx.DefaultPosition, wx.DefaultSize, image_stream_combo_boxChoices, wx.CB_READONLY )
		self.image_stream_combo_box.SetSelection( 0 )
		bSizer191.Add( self.image_stream_combo_box, 0, wx.ALL|wx.EXPAND, 5 )


		self.image_stream_panel.SetSizer( bSizer191 )
		self.image_stream_panel.Layout()
		bSizer191.Fit( self.image_stream_panel )
		bSizer20.Add( self.image_stream_panel, 0, wx.EXPAND, 5 )

		self.m_panel11 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer11 = wx.BoxSizer( wx.VERTICAL )

		self.m_staticText1421 = wx.StaticText( self.m_panel11, wx.ID_ANY, u"Histogram", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1421.Wrap( -1 )
		self.m_staticText1421.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

		bSizer11.Add( self.m_staticText1421, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5 )

		self.histogram_panel = wx.Panel( self.m_panel11, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer11.Add( self.histogram_panel, 1, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5 )


		self.m_panel11.SetSizer( bSizer11 )
		self.m_panel11.Layout()
		bSizer11.Fit( self.m_panel11 )
		bSizer20.Add( self.m_panel11, 1, wx.EXPAND|wx.TOP, 5 )

		self.m_panel23 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer111 = wx.BoxSizer( wx.VERTICAL )

		bSizer12 = wx.BoxSizer( wx.HORIZONTAL )

		self.m_button2 = wx.Button( self.m_panel23, wx.ID_ANY, u"Toggle", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer12.Add( self.m_button2, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.m_staticText7 = wx.StaticText( self.m_panel23, wx.ID_ANY, u"Show saturated pixels in red", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )
		bSizer12.Add( self.m_staticText7, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		bSizer111.Add( bSizer12, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 5 )

		bSizer13 = wx.BoxSizer( wx.VERTICAL )

		bSizer14 = wx.BoxSizer( wx.HORIZONTAL )

		self.ir_contrast_strength_txt_ctrl = wx.TextCtrl( self.m_panel23, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer14.Add( self.ir_contrast_strength_txt_ctrl, 0, wx.ALL, 5 )

		self.m_staticText71 = wx.StaticText( self.m_panel23, wx.ID_ANY, u"IR Contrast Stretch Strength", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText71.Wrap( -1 )
		bSizer14.Add( self.m_staticText71, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		bSizer13.Add( bSizer14, 1, wx.EXPAND, 5 )


		bSizer111.Add( bSizer13, 1, wx.EXPAND, 5 )


		self.m_panel23.SetSizer( bSizer111 )
		self.m_panel23.Layout()
		bSizer111.Fit( self.m_panel23 )
		bSizer20.Add( self.m_panel23, 0, wx.EXPAND|wx.TOP, 5 )

		self.m_panel111 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer20.Add( self.m_panel111, 1, wx.EXPAND|wx.TOP|wx.BOTTOM, 5 )

		self.m_panel7 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		bSizer17 = wx.BoxSizer( wx.VERTICAL )

		self.close_button = wx.Button( self.m_panel7, wx.ID_ANY, u"Close", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		bSizer17.Add( self.close_button, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


		self.m_panel7.SetSizer( bSizer17 )
		self.m_panel7.Layout()
		bSizer17.Fit( self.m_panel7 )
		bSizer20.Add( self.m_panel7, 0, wx.EXPAND|wx.TOP, 5 )


		main_size.Add( bSizer20, 5, wx.ALIGN_CENTER_VERTICAL|wx.EXPAND|wx.ALL, 5 )

		bsizer12 = wx.BoxSizer( wx.VERTICAL )

		self.images_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,-1 ), wx.TAB_TRAVERSAL )
		self.images_panel.SetFont( wx.Font( 9, 70, 90, 90, False, wx.EmptyString ) )

		bSizer16 = wx.BoxSizer( wx.VERTICAL )

		self.m_panel24 = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		left_bsizer = wx.BoxSizer( wx.VERTICAL )

		self.cueing_left_image_title3 = wx.StaticText( self.m_panel24, wx.ID_ANY, u"Click To Set Zoom Location", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.cueing_left_image_title3.Wrap( -1 )
		self.cueing_left_image_title3.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

		left_bsizer.Add( self.cueing_left_image_title3, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

		self.full_view_panel = wx.Panel( self.m_panel24, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		left_bsizer.Add( self.full_view_panel, 5, wx.EXPAND |wx.ALL, 5 )

		self.status_text = wx.StaticText( self.m_panel24, wx.ID_ANY, u"Empty", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
		self.status_text.Wrap( -1 )
		self.status_text.SetFont( wx.Font( 9, 70, 90, 90, False, wx.EmptyString ) )

		left_bsizer.Add( self.status_text, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


		self.m_panel24.SetSizer( left_bsizer )
		self.m_panel24.Layout()
		left_bsizer.Fit( self.m_panel24 )
		bSizer16.Add( self.m_panel24, 3, wx.EXPAND, 5 )

		self.m_panel37 = wx.Panel( self.images_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER|wx.TAB_TRAVERSAL )
		left_bsizer2 = wx.BoxSizer( wx.VERTICAL )

		self.cueing_left_image_title2 = wx.StaticText( self.m_panel37, wx.ID_ANY, u"Zoomed View", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.cueing_left_image_title2.Wrap( -1 )
		self.cueing_left_image_title2.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )

		left_bsizer2.Add( self.cueing_left_image_title2, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

		bSizer10 = wx.BoxSizer( wx.HORIZONTAL )

		self.zoom_slider = wx.Slider( self.m_panel37, wx.ID_ANY, 100, 2, 500, wx.DefaultPosition, wx.DefaultSize, wx.SL_BOTTOM|wx.SL_INVERSE|wx.SL_LABELS|wx.SL_VERTICAL|wx.RAISED_BORDER )
		bSizer10.Add( self.zoom_slider, 0, wx.ALL|wx.EXPAND, 5 )

		self.zoomed_view_panel = wx.Panel( self.m_panel37, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer10.Add( self.zoomed_view_panel, 1, wx.ALL|wx.EXPAND, 5 )


		left_bsizer2.Add( bSizer10, 1, wx.EXPAND, 5 )


		self.m_panel37.SetSizer( left_bsizer2 )
		self.m_panel37.Layout()
		left_bsizer2.Fit( self.m_panel37 )
		bSizer16.Add( self.m_panel37, 3, wx.EXPAND, 5 )


		self.images_panel.SetSizer( bSizer16 )
		self.images_panel.Layout()
		bSizer16.Fit( self.images_panel )
		bsizer12.Add( self.images_panel, 1, wx.EXPAND|wx.TOP|wx.BOTTOM|wx.RIGHT, 5 )


		main_size.Add( bsizer12, 10, wx.EXPAND, 5 )


		self.SetSizer( main_size )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.image_stream_combo_box.Bind( wx.EVT_COMBOBOX, self.on_select_stream )
		self.m_button2.Bind( wx.EVT_BUTTON, self.on_toggle_saturated_pixels )
		self.ir_contrast_strength_txt_ctrl.Bind( wx.EVT_TEXT, self.on_ir_contrast_strength )
		self.close_button.Bind( wx.EVT_BUTTON, self.on_close_button )
		self.zoom_slider.Bind( wx.EVT_SCROLL, self.test )

	def __del__( self ):
		# Disconnect Events
		self.image_stream_combo_box.Unbind( wx.EVT_COMBOBOX, None )
		self.m_button2.Unbind( wx.EVT_BUTTON, None )
		self.ir_contrast_strength_txt_ctrl.Unbind( wx.EVT_TEXT, None )
		self.close_button.Unbind( wx.EVT_BUTTON, None )
		self.zoom_slider.Unbind( wx.EVT_SCROLL, None )


	# Virtual event handlers, overide them in your derived class
	def on_select_stream( self, event ):
		event.Skip()

	def on_toggle_saturated_pixels( self, event ):
		event.Skip()

	def on_ir_contrast_strength( self, event ):
		event.Skip()

	def on_close_button( self, event ):
		event.Skip()

	def test( self, event ):
		event.Skip()


