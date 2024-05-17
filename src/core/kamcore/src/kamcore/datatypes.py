from six import string_types

def render_inner_str(s_or_obj, quote='"'):
    if isinstance(s_or_obj, string_types):
        return quote + s_or_obj + quote
    return str(s_or_obj)

class AttrPrinter(object):
    def slot_items(self):
        items = []
        for k in self.__slots__:
            try:
                items.append((k, getattr(self, k)))
            except AttributeError:
                pass

        return items

    def __repr__(self):
        return str(self)

    def __str__(self):
        kves = ["{}={}".format(k, render_inner_str(v)) for k, v in self.slot_items()]
        kwargstr = ", ".join(kves)
        return "{}({})".format(self.__class__.__name__, kwargstr)


class LitePDict(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __str__(self):
        kves = ["{}={}".format(k, render_inner_str(v)) for k, v in self.items()]
        kwargstr = ", ".join(kves)
        return "{}({})".format(self.__class__.__name__, kwargstr)

    def __repr__(self):
        return str(self)

    def keys(self):
        return self.__slots__

    def items(self):
        return [(k, getattr(self, k)) for k in self.__slots__]

    def values(self):
        # convenience method, not optimal
        return [el[1] for el in self.items()]

    @classmethod
    def from_mapping(cls, d):
        tmp = cls.__new__(cls)  # we need a totally blank object
        for key, value in d.items():
            tmp[key] = value
        return tmp

    def copy(self):
        return self.__class__.from_mapping(self)


class ToDictMxn(LitePDict):
    def to_dict(self):
        kvs = []
        for el in self.__slots__:
            if el[0] == "_":
                continue
            try:
                val = getattr(self, el)
                if isinstance(val, ToDictMxn):
                    val = val.to_dict()
                pair = (el, val)
            except AttributeError:
                pair = None
            if pair is not None:
                kvs.append(pair)

        return dict(kvs)


class TryIntoAttrMxn(object):
    def into(self, other_obj_or_cls):
        # type: (type) -> Any
        if isinstance(other_obj_or_cls, type):
            out = other_obj_or_cls()
        else:
            out = other_obj_or_cls

        for el in self.__slots__:
            if el[0] == "_":
                continue
            val = getattr(self, el)
            if isinstance(val, TryIntoAttrMxn):
                val.into(getattr(out, el))
            else:
                setattr(out, el, getattr(self, el))
        return out


class DefaultInitializer(object):
    def __init__(self, **kwargs):
        for dk, dv in self.__defaults__.items():
            if dk not in kwargs:
                kwargs[dk] = dv

        for k, v in kwargs.items():
            setattr(self, k, v)


class ManditoryInitializer(object):
    def __init__(self, **kwargs):
        sslots = set(self.__slots__)
        diffs = sslots.symmetric_difference(kwargs)
        if diffs:
            raise ValueError("missing fields: {}".format([x for x in diffs]))

        for k, v in kwargs.items():
            setattr(self, k, v)


class _Frob(ToDictMxn, TryIntoAttrMxn, ManditoryInitializer):
    __slots__ = ["foo", "bar"]


class _Stamp(ToDictMxn, TryIntoAttrMxn, DefaultInitializer):
    __defaults__ = {"secs": 0, "nsecs": 0}
    __slots__ = __defaults__.keys()


class _Header(ToDictMxn, TryIntoAttrMxn, DefaultInitializer):
    __defaults__ = {"seq": 0, "frame_id": "", "stamp": _Stamp()}
    __slots__ = __defaults__.keys()
