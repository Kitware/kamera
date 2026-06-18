GUIs implemented with wxPython
==============================

This package provides the KAMERA system control panel and related GUI
components.

Source Tree Layout
==================

launch/
-------
ROS launch files for the GUI nodes (e.g. ``system_control_panel.launch``).

scripts/
--------
ROS node entry points (e.g. ``system_control_panel_node.py``).

src/wxpython_gui/
-----------------
Python modules for the GUI. The main window lives under
``system_control_panel/``: ``gui.fbp`` is the wxFormBuilder layout,
``form_builder_output*.py`` are the generated frame classes, and
``gui.py`` subclasses them with application logic. Shared helpers
(e.g. ``RemoteImagePanel.py``) sit alongside
that directory.

Notes
=====
- The wx.Frame loop and ROS callbacks run on different threads. When a
  method of the wx.Frame is used as a ROS callback, do not modify frame
  attributes directly; use ``wx.CallAfter`` to schedule updates on the
  GUI thread. By convention, ROS-facing methods use the ``_ros`` suffix.
- For ROS image callbacks, if new images arrive faster than the frame
  can repaint, the GUI can crash. Only update images when the frame is
  idle.
