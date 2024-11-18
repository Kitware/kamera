import sys
import shapefile
import wx
import wxpython_gui.system_control_panel.form_builder_output_collection_mode as form_builder_output_collection_mode
from wxpython_gui.cfg import SYS_CFG, redis

class CollectionModeFrame(form_builder_output_collection_mode.MainFrame):
    """.

    """
    def __init__(self, parent, collection_mode, shapefile_fname,
                 use_archive_region, allow_ir_nuc,
                 trigger_freq, overlap_percent):
        """
        :param collection_mode:
        :type collection_mode: str

        :param shapefile_fname: Currently loaded shapefile, or None
        :type shapefile_fname: Optional[str]

        :param use_archive_region: If selected shapefile should be "used" or not
        :type use_archive_region: Optional[bool]
        """
        form_builder_output_collection_mode.MainFrame.__init__(self, parent)
        self.parent = parent
        self.shapefile_fname = shapefile_fname
        self.use_archive_region = int(use_archive_region)
        self.allow_ir_nuc = int(allow_ir_nuc)
        self.trigger_freq = trigger_freq
        self.overlap_percent = overlap_percent
        self.update_options(collection_mode)
        self.Show()
        self.SetMinSize(self.GetSize())

    def update_options(self, collection_mode):
        if collection_mode == "fixed rate":
            self.mode_combo_box.SetSelection(0)
            self.on_set_mode("nomode")
            try:
                collection_value = self.trigger_freq
                self.rate_txtctrl.SetValue(str(collection_value))
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                msg = "{}:{}\n{}: {}".format(fname, exc_tb.tb_lineno, exc_type.__name__, e)
                print(msg)

        else:
            if collection_mode == "fixed overlap":
                collection_value = self.overlap_percent
                self.mode_combo_box.SetSelection(1)
                self.on_set_mode("nomode")
                try:
                    self.overlap_txtctrl.SetValue(str(collection_value))
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    msg = "{}:{}\n{}: {}".format(fname, exc_tb.tb_lineno, exc_type.__name__, e)
                    print(msg)

            else:
                self.save_button.Show()
                self.percent_panel.Show()
                self.rate_txtctrl.Hide()
                self.self.m_staticText4.Hide()
        if self.shapefile_fname != "":
            self.shapefile_file_picker.SetPath(str(self.shapefile_fname))
            self.shapefile_file_picker.path = str(self.shapefile_fname)
        self.shapefile_checkbox.SetValue(self.use_archive_region == 1)
        self.allow_nuc.SetValue(self.allow_ir_nuc == 1)


    def on_set_mode(self, event=None):
        ind = self.mode_combo_box.GetCurrentSelection()
        mode = self.mode_combo_box.GetStringSelection()
        if event != "nomode":
            self.update_options(mode)

        if ind == -1:
            self.save_button.Hide()
            self.percent_panel.Hide()
            self.rate_txtctrl.Hide()
            self.m_staticText4.Hide()
        elif ind == 0:
            self.rate_txtctrl.Show()
            self.m_staticText4.Show()
            self.save_button.Show()
            self.percent_panel.Hide()
        elif ind == 1:
            self.rate_txtctrl.Hide()
            self.m_staticText4.Hide()
            self.save_button.Show()
            self.percent_panel.Show()

        self.percent_panel.GetParent().Layout()

    def on_select_shapefile(self, event):
        self.shapefile_fname = self.shapefile_file_picker.GetPath()
        if self.is_valid_shapefile(self.shapefile_fname):
            self.shapefile_checkbox.SetValue(True)
        else:
            self.shapefile_fname = ""
            self.shapefile_file_picker.SetPath(str(self.shapefile_fname))
            self.shapefile_file_picker.path = str(self.shapefile_fname)

    def on_save(self, event):
        """When the 'Save' button is selected.

        """
        ind = self.mode_combo_box.GetCurrentSelection()
        err_msg = 'Must enter a value number'
        try:
            if ind == 0:
                mode = 'fixed rate'
                value = float(self.rate_txtctrl.GetValue())
                if value < SYS_CFG["arch"]["min_frame_rate"] or value > SYS_CFG["arch"]["max_frame_rate"]:
                    err_msg = 'Frame rate must be in range {}-{} frames/second'.format(
                            SYS_CFG["arch"]["min_frame_rate"], SYS_CFG["arch"]["max_frame_rate"])
                    raise ValueError
                self.collection_mode = mode
                self.trigger_freq = value
            elif ind == 1:
                mode = 'fixed overlap'
                value = float(self.overlap_txtctrl.GetValue())
                if value > 99:
                    err_msg = 'Image overlap must be < 100 percent.'
                    raise ValueError
                self.collection_mode = mode
                self.overlap_percent = value
        except ValueError:
            dlg = wx.MessageDialog(self, err_msg, 'Error',
                                   wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return

        if self.allow_nuc.IsChecked():
            self.allow_ir_nuc = 1
        else:
            self.allow_ir_nuc = 0

        if self.shapefile_checkbox.IsChecked():
            if self.shapefile_fname != "":
                self.use_archive_region = 1
                self.set_collect_in_region(self.shapefile_fname)
            else:
                msg = 'Must select valid shapefile if checkbox is checked.'
                dlg = wx.MessageDialog(self, msg, 'Error', wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
        else:
            self.use_archive_region = 0

        # Hold state so last selection is remembered, even if process is canceled
        print("Setting SYS_CFG configs")
        SYS_CFG['collection_mode'] = self.collection_mode
        SYS_CFG["shapefile_fname"] = self.shapefile_fname
        SYS_CFG["arch"]["use_archive_region"] = self.use_archive_region
        SYS_CFG["arch"]["allow_ir_nuc"] = self.allow_ir_nuc
        print("Changing trigger freq to %s" % self.trigger_freq)
        s = "saving collection_mode, state is: "
        if self.collection_mode == "fixed overlap":
            SYS_CFG["arch"]["overlap_percent"] = self.overlap_percent
            SYS_CFG["arch"]["load_shapefile"] = self.use_archive_region
            s += "overlap_percent: %s | " % self.overlap_percent
            s += "shapefile_fname: %s | " % self.shapefile_fname
        else:
            SYS_CFG["arch"]["trigger_freq"] = self.trigger_freq
            s += "trigger_freq: %s | " % self.trigger_freq
        s += "collection_mode: %s | " % self.collection_mode
        s += "allow IR NUCing: %s | " % bool(self.allow_ir_nuc)
        s += "use_archive_region: %s | " % bool(self.use_archive_region)
        self.parent.add_to_event_log(s)
        self.Close()

    def is_valid_shapefile(self, fname):
        print("Trying to load shapefile: %s" % fname)
        try:
            sf = shapefile.Reader(fname)
        except:
            msg = 'Could not open shapefile %s.' % fname
            dlg = wx.MessageDialog(self, msg, 'Error', wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return False
        return True

    def set_collect_in_region(self, fname):
        """Load shapefile 'fname' to be used for collection region trigger.

        """
        # Dump into redis db for shapefile node to read
        if self.is_valid_shapefile(fname):
            with open(fname, 'rb') as f:
            # Dump into redis db for shapefile node to read
                b = f.read()
                # Only place we have to set the shapefile, since it requires
                # the redis handle
                redis.set("/data/shapefile", b)
        else:
            return False
        return True

    def on_cancel(self, event=None):
        """When the 'Cancel' button is selected.

        """
        self.Close()

