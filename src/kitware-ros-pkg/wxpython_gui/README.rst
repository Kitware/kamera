GUIs implemented with wxPython
==============================

This package provides various GUIs implemented with wxPython.

Source Tree Layout
==================

launch/
-------
ROS launch files launching the node or nodes in /scripts.

resources/
----------
Any data (e.g., images) required for the GUIs.

scripts/
--------
The ROS node launcher for each GUI is defined with a Python script here.

src/wxpython_gui/
-----------------
The current convention is that each module under `src/wxpython_gui` defines one GUI. Each folder contains a `gui.py`, where the wx main loop is defined. However, the general layout of the GUI is defined using wxFormBuilder (`necessary version <https://sourceforge.net/projects/wxformbuilder/files/wxformbuilder-nightly/3.4.2-beta/>`_, see install instructions for Ubuntu), stored in `gui.fbp`. wxFormBuilder automatically generates the code `form_builder_output.py`, which defines MainFrame, a subclass of wx.Frame, setting up the layout. In `gui.py`, the MainFrame of the gui is subclassed from `form_builder_output.MainFrame`, absorbing all of the layout definition. Then additional processing is defined within gui.py to complete the GUI functionality.

src/wxpython_gui/wx_elements.py
-------------------------------
Common GUI functionality. Currently, the ImagePanelManager is the only object
defined (see documentation therein).

Notes
=====
- The wx.Frame loop and ROS callbacks will run on different threads. Therefore, when a method of the wx.Frame is provided as a ROS callback, attributes of the wx.Frame should not be modified directly but rather wx.CallAfter should be used to call another method to make the modifications. By convention, any method that interfaces with ROS should have the subscript '_ros'.
- For ROS callbacks providing new images, if the rate at which new images arrive exceeds the rate at which the wx.Frame can handle updating images, the frame will crash. Therefore, a catch should be used to only try to update images if the frame is idle.
