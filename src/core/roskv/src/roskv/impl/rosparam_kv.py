import rospy
from roskv.base import KV, NullDefault

_default = NullDefault()


class RosParamKV(KV):
    def __init__(self):
        pass

    def get(self, key, default=_default, **kwargs):
        if _default is _default:
            return rospy.get_param(key)
        return rospy.get_param(key, default)

    def put(self, key, val, **kwargs):
        return rospy.set_param(key, val)

    def delete(self, key, **kwargs):
        return rospy.delete_param(key)
