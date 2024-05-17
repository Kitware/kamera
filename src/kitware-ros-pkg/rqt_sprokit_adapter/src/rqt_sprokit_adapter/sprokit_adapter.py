import os
import argparse
import rospy
import rospkg
import distutils.util

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtGui import QWidget
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

class SprokitAdapterTopicHandler(object):
    def __init__(self, topic):
        self.topic = topic
        self._publisher = rospy.Publisher(topic,DiagnosticStatus)

    def publish(self,showProbability=True,threshold=True):
        msg = DiagnosticStatus()
        msg.name = "SprokitAdapter"
        msg.level = DiagnosticStatus.OK
        msg.hardware_id = self.topic
        msg.message =  "No Message"
        msg.values = [
                KeyValue(key='threshold',value=str(threshold)),
                KeyValue(key='draw_text',value=str(showProbability))
                    ]
        if self._publisher is not None:
            self._publisher.publish(msg)

    def close(self):
        self._publisher.unregister()
        del self._publisher
        self._publisher = None
        self._topic = None

class SprokitAdapter(Plugin):

    def __init__(self, context):
        super(SprokitAdapter, self).__init__(context)
        self.initialized = False
        self._topicHandler = None
        # Give QObjects reasonable names
        self.setObjectName('SprokitAdapter')

        args = self._parse_args(context.argv())

        # Create QWidget
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_sprokit_adapter'), 'resource', 'rqt_sprokit_adapter.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)

        # Give QObjects reasonable names
        self._widget.setObjectName('SprokitAdapterUi')

        # Show _widget.windowTitle on left-top of each plugin (when 
        # it's set in _widget). This is useful when you open multiple 
        # plugins at once. Also if you open multiple instances of your 
        # plugin at once, these lines add number to make it easy to 
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))

        # Add widget to the user interface
        context.add_widget(self._widget)

        self._widget.topicLineEdit.editingFinished.connect(self._publish)
        self._widget.showProbabilityCheckBox.clicked.connect(self._publish)
        self._widget.thresholdSpinBox.valueChanged.connect(self._publish)

    def shutdown_plugin(self):
        if self._topicHandler is not None:
            self._topicHandler.close()
            self._topicHandler = None

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        instance_settings.set_value('topic',self._widget.topicLineEdit.text())
        instance_settings.set_value('threshold',self._widget.thresholdSpinBox.value())
        instance_settings.set_value('draw_text',self._widget.showProbabilityCheckBox.isChecked())

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        topic = instance_settings.value('topic')
        print("Topic ",topic)
        threshold = instance_settings.value('threshold')
        print("Threshold ",threshold)
        draw_text = instance_settings.value('draw_text')
        print("draw text ",draw_text)
        self._widget.topicLineEdit.setText(topic)
        self._widget.thresholdSpinBox.setValue(float(threshold))
        self._widget.showProbabilityCheckBox.setCheckState(distutils.util.strtobool(draw_text))
        self.initialized = True
        self._publish()
        pass

    def _parse_args(self, argv):
        parser = argparse.ArgumentParser(prog='rqt_bag', add_help=False)
        SprokitAdapter.add_arguments(parser)
        return parser.parse_args(argv)

    def _publish(self):
        #print("Topic: ",self._widget.topicLineEdit.text())
        #print("ShowText: ",self._widget.showProbabilityCheckBox.isChecked())
        #print("Threshold ",self._widget.thresholdSpinBox.value())
        topic = self._widget.topicLineEdit.text()
        if topic == "":
            return
        if not self._topicHandler or self._topicHandler.topic != topic:
            if not self._topicHandler is None:
                self._topicHandler.close()
            self._topicHandler = SprokitAdapterTopicHandler(topic)

        self._topicHandler.publish(self._widget.showProbabilityCheckBox.isChecked(),
                                   self._widget.thresholdSpinBox.value()
                            )
    

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Options for rqt_sprokit_adapter plugin')
        group.add_argument('--topic', default='display_parameters', help='publish the clock time')

    #def trigger_configuration(self):
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog
