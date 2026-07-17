import wx
import wxpython_gui.system_control_panel.form_builder_output_event_log_note as form_builder_output_event_log_note
from wxpython_gui.system_control_panel.gui_utils import unclip_static_text

class EventLogNoteFrame(form_builder_output_event_log_note.MainFrame):
    """.

    """
    def __init__(self, parent):
        """

        """
        # Initialize parent class
        form_builder_output_event_log_note.MainFrame.__init__(self, parent)
        # Recompute enlarged title fonts so they are not clipped under Phoenix.
        wx.CallAfter(unclip_static_text, self)
        self.parent = parent
        self.Show()
        self.SetMinSize(self.GetSize())

    def on_save(self, event):
        """When the 'Save' button is selected.

        """
        event_type = self.event_type_combo.GetStringSelection()
        if event_type == '':
            return
        event_note = self.note_textCtrl.GetValue()
        self.parent.add_to_event_log('%s: %s' % (event_type, event_note))

        self.Close()

    def on_cancel(self, event=None):
        """When the 'Cancel' button is selected.

        """
        self.Close()

