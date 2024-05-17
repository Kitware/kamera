from six import PY2, PY3

if PY2:
    JSONDecodeError = ValueError

elif PY3:
    from json import JSONDecodeError

else:
    raise RuntimeError("Unknown python version")
