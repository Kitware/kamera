from __future__ import division, print_function
import numpy as np
from wxpython_gui.UpdateImageThread import UpdateImageThread

class UpdateImageThreadFit(UpdateImageThread):
    """Request imagery to be fit to the panel.

    """
    def __init__(self, parent, srv_topic,
                 max_mpix, compressed=False):
        """
        :param srv_topic: ROS topic for RequestImageView service to provide
            the required imagery.
        :type srv_topic: str

        """
        # Initialize parent class
        super(UpdateImageThreadFit, self).__init__(parent, srv_topic,
                                                   compressed=compressed)
        self.max_mpix = max_mpix

    def get_homography(self, preview=False):
        """Return homography to warp from panel to raw-image coordinates.

        """
        panel_width, panel_height = self._parent.wx_panel.GetSize()

        # Clamp to maximum requested size.
        s = self.max_mpix/panel_width*panel_height
        if s < 1:
            s = np.sqrt(s)
            panel_height = int(np.round(panel_height*s))
            panel_width = int(np.round(panel_width*s))

        if preview:
            im_height, im_width = 480, 640
        else:
            im_height = self._raw_image_height
            im_width = self._raw_image_width
        s = min(panel_height/im_height, panel_width/im_width)
        homography = np.array([[s,0,0],[0,s,0],[0,0,1]])

        output_height = np.floor(im_height*s)
        output_width = np.floor(im_width*s)
        return homography, output_height, output_width
