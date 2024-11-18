from __future__ import division, print_function
import numpy as np
from wxpython_gui.UpdateImageThread import UpdateImageThread

class UpdateImageThreadZoom(UpdateImageThread):
    """Request imagery to be fit to the panel.

    """
    def __init__(self, parent, srv_topic, center_callback,
                 zoom_callback, max_mpix, compressed=False):
        """
        :param srv_topic: ROS topic for RequestImageView service to provide
            the imagery needed.
        :type srv_topic: str


        """
        # Initialize parent class
        super(UpdateImageThreadZoom, self).__init__(parent, srv_topic,
                                                    compressed=compressed)
        self.max_mpix = max_mpix
        self.center_callback = center_callback
        self.zoom_callback = zoom_callback
        self.sub = False # overrwrite image subscription, so we request image views

    def get_homography(self, preview=False):
        """Return homography to warp from panel to raw-image coordinates.

        """
        #print('on_size')
        panel_width, panel_height = self._parent.wx_panel.GetSize()

        # Clamp to maximum requested size.
        s = self.max_mpix/(panel_width*panel_height)
        if s < 1:
            s = np.sqrt(s)
            panel_height = int(np.round(panel_height*s))
            panel_width = int(np.round(panel_width*s))

        s = self.zoom_callback()/100
        center = self.center_callback()
        if center is None:
            if self._raw_image_height is not None and \
               self._raw_image_width is not None:
                center = (self._raw_image_height/2,self._raw_image_width/2)

        tx = panel_width/2-s*center[0]
        ty = panel_height/2-s*center[1]
        homography = np.array([[s,0,tx],[0,s,ty],[0,0,1]])
        return homography, panel_height, panel_width
