from . import base
from custom_msgs.msg import ArchiveSchema

class ArchiveSchemaDispatch(base.DispatchBase):
    message_class = ArchiveSchema
    pubs = {}
    __slots__ = 'project', 'flight'
    def __new__(cls, **kwargs):

        self = object.__new__(cls)
        self.msg = self.new_message()

        self.msg.project = kwargs.get('project', 'default2019')
        self.msg.flight  = kwargs.get('flight', 'fl00')
        return self

aa = ArchiveSchemaDispatch()
