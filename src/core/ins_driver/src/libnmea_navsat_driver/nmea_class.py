import abc
import rospy


class NMEA(object):
    def __init__(self, name, msg, queue_size=1):
        self._name = name
        self._queue_size = queue_size
        self._msg = msg
        self._pub = rospy.Publisher(name, msg, queue_size=queue_size)

    @abc.abstractmethod
    def from_dict(self, data):
        # type: (dict) -> genpy.msg
        """
        This is just an example and should be overridder.
        Args:
            data: dict to be parsed

        Returns:
            Rospy message populated from dict
        """
        msg = self._msg()
        msg.header.stamp = data['current_time']
        return msg

    def publish_from_dict(self, data):
        # type: (dict) -> bool
        msg = self.from_dict(data)
        if msg is None:
            return False
        self.publish(msg)
        return True

    @staticmethod
    def format_header(msg, data_or_header):
        # type: (genpy.msg, dict) -> None
        """
        Populate the message with header information.
        Try to extract a field 'header' from the dict. Upon failing,
        treat the dict as the header structure itself.
        Args:
            msg:
            data_or_header:

        Returns:

        """
        maybe_header = data_or_header.get('header', None)
        if maybe_header is None:
            header = data_or_header
        else:
            header = maybe_header
        msg.header.stamp = header['stamp']
        msg.header.frame_id = header['frame_id']

    def msg_from_header(self, data_or_header):
        msg = self.msg()
        self.format_header(msg, data_or_header)
        return msg

    @property
    def name(self):
        return self._name

    @property
    def msg(self):
        return self._msg

    def publish(self, msg):
        self._pub.publish(msg)
