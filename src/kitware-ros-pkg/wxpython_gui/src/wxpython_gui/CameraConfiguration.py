from __future__ import division, print_function
import copy
import wx
import yaml
import numpy as np
import sys
from wxpython_gui.camera_models import load_from_file
import wxpython_gui.system_control_panel.form_builder_output_camera_configuration as fbocc
from wxpython_gui.cfg import SYS_CFG, save_camera_config

class CameraConfiguration(fbocc.MainFrame):
    """Defines how the cameras are layed out.

    """
    def __init__(self, parent,
                 sel_camera_config):
        # type: (Any, dict, wx.ComboBox, Callable, str) -> None
        """
        :param parent: Parent.
        :type parent: wx object

        """
        # Initialize parent class
        fbocc.MainFrame.__init__(self, parent)
        self.parent = parent
        self.curr_cfg = sel_camera_config

        self.pipe_wildcard = 'Pipefile (*.pipe)|*.pipe|All Files (*)|*'
        self.yaml_wildcard = 'Camera Model (*.yaml)|*.yaml|All Files (*)|*'

        self.update_combo(select_str=sel_camera_config)

    def get_template(self):
        tmpl = {}
        tmpl['left_sys_pipe'] = ""
        tmpl['left_rgb_yaml_path'] = ""
        tmpl['left_ir_yaml_path'] = ""
        tmpl['left_uv_yaml_path'] = ""

        tmpl['center_sys_pipe'] = ""
        tmpl['center_rgb_yaml_path'] = ""
        tmpl['center_ir_yaml_path'] = ""
        tmpl['center_uv_yaml_path'] = ""

        tmpl['right_sys_pipe'] = ""
        tmpl['right_rgb_yaml_path'] = ""
        tmpl['right_ir_yaml_path'] = ""
        tmpl['right_uv_yaml_path'] = ""

        tmpl['description'] = ""

        return tmpl

    def update_combo(self, select_str=""):
        """ Update 'camera_config_combo' to respect 'camera_configuration_dict'.
            select_str - select this str after updating.
        """
        # First cache which camera configuration is currently selected.
        if select_str is "":
            select_str = self.camera_config_combo.GetStringSelection()

        self.camera_config_combo.SetEditable(True)
        self.camera_config_combo.Clear()

        ind = None
        n = 0
        keys = sorted(SYS_CFG["camera_cfgs"].keys(), reverse=True)
        if len(keys) > 0:
            for camera_config_combo in keys:
                if camera_config_combo == select_str:
                    ind = n

                self.camera_config_combo.Append(camera_config_combo)
                n += 1

        self.camera_config_combo.SetEditable(False)

        if ind is not None:
            self.camera_config_combo.SetSelection(ind)
        else:
            self.camera_config_combo.SetSelection(0)

        # The above SetSelection doesn't trigger 'on_combo_select' by itself.
        self.on_combo_select()

    def on_save(self, event):
        """Save the current panel's updates.

        """
        key = self.config_name_txt_ctrl.GetValue()

        if key == '':
            dlg = wx.MessageDialog(self,
                                   'System Configuration Name cannot be empty',
                                   'Error', wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Show()
            return

        # Sanitize name.
        key = key.replace(' ', '_')
        key = key.replace('.', '_')

        tmp = {}
        tmp['left_sys_pipe'] = self.left_pipe_picker.GetValue()
        tmp['left_rgb_yaml_path'] = self.left_rgb_yaml_picker.GetValue()
        tmp['left_ir_yaml_path'] = self.left_ir_yaml_picker.GetValue()
        tmp['left_uv_yaml_path'] = self.left_uv_yaml_picker.GetValue()

        tmp['center_sys_pipe'] = self.center_pipe_picker.GetValue()
        tmp['center_rgb_yaml_path'] = self.center_rgb_yaml_picker.GetValue()
        tmp['center_ir_yaml_path'] = self.center_ir_yaml_picker.GetValue()
        tmp['center_uv_yaml_path'] = self.center_uv_yaml_picker.GetValue()

        tmp['right_sys_pipe'] = self.right_pipe_picker.GetValue()
        tmp['right_rgb_yaml_path'] = self.right_rgb_yaml_picker.GetValue()
        tmp['right_ir_yaml_path'] = self.right_ir_yaml_picker.GetValue()
        tmp['right_uv_yaml_path'] = self.right_uv_yaml_picker.GetValue()

        tmp['description'] = self.configuration_notes_txt_ctrl.GetValue()


        SYS_CFG["camera_cfgs"][key] = tmp

        self.curr_cfg = key
        self.update_combo(select_str=key)
        # Refresh current combo in parent in case it changed
        self.parent.on_camera_config_combo()

        # Remove a tmp definition if present.
        try:
            del SYS_CFG["camera_cfgs"]['']
        except KeyError:
            pass

        self.save_camera_config_dict(SYS_CFG["camera_cfgs"])

    def save_camera_config_dict(self, config_dict):
        # Save camera config to only local config here, saved to /mnt in parent
        dirname = save_camera_config()
        self.parent.add_to_event_log('Saved system configurations to {}. '
                .format(dirname))

    def on_combo_select(self, event=None):
        select_str = self.camera_config_combo.GetStringSelection()
        try:
            tmp = SYS_CFG["camera_cfgs"][select_str]
        except KeyError:
            return

        self.set_fields_to_camera_config_dict(tmp)
        self.config_name_txt_ctrl.SetValue(select_str)
        self.config_name_txt_ctrl.Disable()
        self.curr_cfg = select_str

    def set_fields_to_camera_config_dict(self, config_dict):
        """Update all of the path txt str in the frame.

        """
        txt = config_dict['left_sys_pipe']
        self.left_pipe_picker.SetValue('' if txt is None else txt)

        txt = config_dict['left_rgb_yaml_path']
        self.left_rgb_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['left_ir_yaml_path']
        self.left_ir_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['left_uv_yaml_path']
        self.left_uv_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['center_sys_pipe']
        self.center_pipe_picker.SetValue('' if txt is None else txt)

        txt = config_dict['center_rgb_yaml_path']
        self.center_rgb_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['center_rgb_yaml_path']
        self.center_rgb_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['center_ir_yaml_path']
        self.center_ir_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['center_uv_yaml_path']
        self.center_uv_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['right_sys_pipe']
        self.right_pipe_picker.SetValue('' if txt is None else txt)

        txt = config_dict['right_rgb_yaml_path']
        self.right_rgb_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['right_ir_yaml_path']
        self.right_ir_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['right_uv_yaml_path']
        self.right_uv_yaml_picker.SetValue('' if txt is None else txt)

        txt = config_dict['description']
        self.configuration_notes_txt_ctrl.SetValue('' if txt is None else txt)

    def on_new(self, event):
        self.config_name_txt_ctrl.Enable()
        tmpl = self.get_template()
        self.set_fields_to_camera_config_dict(tmpl)
        self.config_name_txt_ctrl.SetValue('')
        self.camera_config_combo.SetSelection(-1)

    def on_new_from_current(self, event):
        self.config_name_txt_ctrl.Enable()
        self.config_name_txt_ctrl.SetValue('')

    # ----------------------------- File Pickers -----------------------------
    # Left-system pickers
    def on_find_left_rgb_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.left_rgb_yaml_picker.SetValue(file_path)

    def on_clear_left_rgb_yaml(self, event):
        self.left_rgb_yaml_picker.SetValue('')

    def on_find_left_ir_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.left_ir_yaml_picker.SetValue(file_path)

    def on_clear_left_ir_yaml(self, event):
        self.left_ir_yaml_picker.SetValue('')

    def on_find_left_uv_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.left_uv_yaml_picker.SetValue(file_path)

    def on_clear_left_uv_yaml(self, event):
        self.left_uv_yaml_picker.SetValue('')

    def on_find_left_pipe(self, event):
        file_path = self.file_picker(wildcard=self.pipe_wildcard)
        self.left_pipe_picker.SetValue(file_path)

    def on_clear_left_pipe(self, event):
        self.left_pipe_picker.SetValue('')

    # Center-system pickers
    def on_find_center_rgb_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.center_rgb_yaml_picker.SetValue(file_path)

    def on_clear_center_rgb_yaml(self, event):
        self.center_rgb_yaml_picker.SetValue('')

    def on_find_center_ir_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.center_ir_yaml_picker.SetValue(file_path)

    def on_clear_center_ir_yaml(self, event):
        self.center_ir_yaml_picker.SetValue('')

    def on_find_center_uv_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.center_uv_yaml_picker.SetValue(file_path)

    def on_clear_center_uv_yaml(self, event):
        self.center_uv_yaml_picker.SetValue('')

    def on_find_center_pipe(self, event):
        file_path = self.file_picker(wildcard=self.pipe_wildcard)
        self.center_pipe_picker.SetValue(file_path)

    def on_clear_center_pipe(self, event):
        self.center_pipe_picker.SetValue('')

    # Right-system pickers
    def on_find_right_rgb_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.right_rgb_yaml_picker.SetValue(file_path)

    def on_clear_right_rgb_yaml(self, event):
        self.right_rgb_yaml_picker.SetValue('')

    def on_find_right_ir_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.right_ir_yaml_picker.SetValue(file_path)

    def on_clear_right_ir_yaml(self, event):
        self.right_ir_yaml_picker.SetValue('')

    def on_find_right_uv_yaml(self, event):
        file_path = self.file_picker(wildcard=self.yaml_wildcard)
        self.right_uv_yaml_picker.SetValue(file_path)

    def on_clear_right_uv_yaml(self, event):
        self.right_uv_yaml_picker.SetValue('')

    def on_find_right_pipe(self, event):
        file_path = self.file_picker(wildcard=self.pipe_wildcard)
        self.right_pipe_picker.SetValue(file_path)

    def on_clear_right_pipe(self, event):
        self.right_pipe_picker.SetValue('')

    def file_picker(self, wildcard='*'):
        dialog = wx.FileDialog(None, "Choose a file", SYS_CFG["nas_mnt"], '',
                               wildcard, wx.OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
        else:
            print("Invalid pipe path selected, not setting.")
            return

        return file_path
    # ------------------------------------------------------------------------

    def on_delete(self, event):
        key = self.camera_config_combo.GetStringSelection()
        try:
            del SYS_CFG["camera_cfgs"][key]
        except KeyError:
            return
        ind = self.camera_config_combo.GetSelection()
        ind = min([ind, self.camera_config_combo.GetCount() - 1])
        self.camera_config_combo.SetSelection(ind)
        select_str = self.camera_config_combo.GetStringSelection()
        self.update_combo(select_str)
        self.curr_cfg = select_str
        self.parent.add_to_event_log('Deleted system configuration {}.'
                .format(self.curr_cfg))
        self.save_camera_config_dict(SYS_CFG["camera_cfgs"])
        self.parent.set_camera_config_dict()

    def on_done(self, event=None):
        """When the 'Cancel' button is selected.

        """
        self.Close()
