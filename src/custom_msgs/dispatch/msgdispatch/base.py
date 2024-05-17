import rospy

# This is an abstract base class
class DispatchBase(object):
    publisher = None
    msg = None

    # todo: better pattern to manage class-bound variables
    # message_class, pubs, and optionally counter should be re-defined in each child
    message_class = None
    pubs = {}
    counter = 0

    @classmethod
    def next_id(cls):
        x = int(cls.counter)
        cls.counter += 1
        return x

    @classmethod
    def add_publisher(cls, name, queue_size=3):
        if name not in cls.pubs:
            cls.pubs.update(
                {'name': rospy.Publisher(name, cls.message_class, tcp_nodelay=True,
                                         queue_size=queue_size)})
        else:
            raise ValueError('Publisher already exists: {}'.format(name))

    @property
    def new_message(self):
        return self.message_class

    def publish(self):
        for pub in self.pubs.values():
            pub.publish(self.msg)

    def __repr__(self):
        return str(self.msg)

    def __getitem__(self, item):
        return getattr(self.msg, item)

    def __getattr__(self, item):
        return getattr(self.msg, item)

    def as_dict_headless(self):
        dd = dict(self.__dict__)
        dd.pop('header', None)
        return dd

    def dump_yml(self, filename):
        with open(filename, 'a') as fp:
            data = str(self.msg) + '\n---\n'
            fp.write(data)
