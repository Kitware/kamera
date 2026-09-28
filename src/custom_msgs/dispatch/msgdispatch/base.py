# This is an abstract base class
class DispatchBase(object):
    msg = None

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
    def add_publisher(cls, node, name, queue_size=3):
        """Register a publisher on `node` for this dispatch class.

        In ROS2 publishers are owned by a node, so one must be provided.
        """
        if name in cls.pubs:
            raise ValueError('Publisher already exists: {}'.format(name))
        cls.pubs[name] = node.create_publisher(cls.message_class, name, queue_size)

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
