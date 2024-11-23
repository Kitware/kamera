import wx
import wxpython_gui.system_control_panel.form_builder_output_effort_metadata as form_builder_output_effort_metadata
from wxpython_gui.cfg import SYS_CFG


class MetadataEntryFrame(form_builder_output_effort_metadata.MainFrame):
    """Metadata entry defining flight NO., effort, project, etc."""

    def __init__(
        self, parent, effort_metadata_dict, effort_combo_box, edit_effort_name=None
    ):
        # type: (Any, dict, wx.ComboBox, Callable, str) -> None
        """
        :param parent: Parent.
        :type parent: wx object

        :param effort_metadata_dict: Dictionary with key being the effort
            nickname and the value being a dictionary with keys 'project_name',
            'aircraft', 'flight', and 'field_notes'.
        :type effort_metadata_dict: dict

        :param effort_combo_box: Combo box for the collection event selection.
        :type effort_combo_box: wx.ComboBox

        :param edit_effort_name: Name of existing event that we want to edit.
        :type edit_effort_name: str

        """
        # Initialize parent class
        form_builder_output_effort_metadata.MainFrame.__init__(self, parent)
        self.parent = parent
        self.effort_metadata_dict = effort_metadata_dict
        self.effort_combo_box = effort_combo_box
        self.edit_effort_name = edit_effort_name

        if edit_effort_name is not None or self.effort_combo_box.GetCount() == 0:
            # We are in a state where we are adding a new entry and we have
            self.on_populate_from_last_entry_button.Hide()
            self.Layout()

        if edit_effort_name is not None and edit_effort_name in effort_metadata_dict:
            self.fill_from_effort_dict(edit_effort_name)
            self.effort_nickname_textCtrl.SetEditable(False)
            self.effort_nickname_textCtrl.SetBackgroundColour((200, 200, 200))

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_up)
        self.Show()
        self.SetMinSize(self.GetSize())
        self.m_staticText3341.Hide()
        self.wait_time_sec.Hide()
        self.m_staticText33411.Hide()
        self.delete_old_images_sec.Hide()

    def on_key_up(self, event):
        keyCode = event.GetKeyCode()
        if keyCode == wx.WXK_ESCAPE:
            self.on_cancel()

        event.Skip()

    def on_populate_from_last_entry(self, event):
        effort_name = self.effort_combo_box.GetStringSelection()
        self.fill_from_effort_dict(effort_name)

    def fill_from_effort_dict(self, effort_name):
        self.effort_nickname_textCtrl.SetValue(effort_name)
        effort_dict = self.effort_metadata_dict[effort_name]
        self.project_name_textCtrl.SetValue(effort_dict["project_name"])
        self.aircraft_textCtrl.SetValue(effort_dict["aircraft"])
        self.field_notes_textCtrl.SetValue(effort_dict["field_notes"])
        self.save_every_x_image.SetValue(str(effort_dict["save_every_x_image"]))
        # unused
        self.wait_time_sec.SetValue(str(1))
        self.delete_old_images_sec.SetValue(str(1))

    def on_save(self, event):
        """When the 'Save' button is selected."""
        effort_name = self.effort_nickname_textCtrl.GetValue()

        # Make sure effort_name is not an empty string.
        if effort_name == "":
            dlg = wx.MessageDialog(
                self, "'Event Name' is empty", "Error", wx.OK | wx.ICON_ERROR
            )
            dlg.ShowModal()
            dlg.Destroy()
            return

        already_exists = False
        num_selections = self.effort_combo_box.GetCount()
        for i in range(num_selections):
            if effort_name == self.effort_combo_box.GetString(i):
                self.effort_combo_box.SetSelection(i)
                already_exists = True
                break

        if already_exists and self.edit_effort_name is None:
            # The effort name already exists and this was not a call to
            # specifically edit the entry.
            dlg = wx.MessageDialog(
                self,
                "Effort name '%s' already exists, "
                "do you want to overwrite the previous "
                "entry with the current one?" % effort_name,
                "Effort Already Exists",
                wx.YES_NO | wx.ICON_QUESTION,
            )
            val = dlg.ShowModal()
            dlg.Show()

            if val == wx.ID_NO:
                dlg.Destroy()
                return

        if not already_exists:
            self.effort_combo_box.Append(effort_name)
            self.effort_combo_box.SetSelection(num_selections)

        effort_dict = {}

        try:
            save_every_x_image = int(self.save_every_x_image.GetValue())
            effort_dict["save_every_x_image"] = save_every_x_image
        except (TypeError, ValueError) as err_msg:
            em = "Must enter value in 'save_every_x_image' field."
            dlg = wx.MessageDialog(self, em, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return

        effort_dict["project_name"] = self.project_name_textCtrl.GetValue()
        effort_dict["aircraft"] = self.aircraft_textCtrl.GetValue()
        effort_dict["field_notes"] = self.field_notes_textCtrl.GetValue()

        self.effort_metadata_dict[effort_name] = effort_dict

        self.Close()

    def on_cancel(self, event=None):
        """When the 'Cancel' button is selected."""
        self.Close()
