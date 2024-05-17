class LowpassIIR(object):
    """
    Digital Infinite impulse response lowpass filter AKA exponential moving
    average. Smooths values.
    """

    def __init__(self, init_state, gamma=0.1,):
        """
        :param gamma: Coefficient for lowpass, (0,1]
        gam=1 -> 100% pass
        """
        self.gamma = gamma
        self._state = init_state

    def update(self, x):
        """
        Push a value into the filter
        :param x: Value of input signal
        :return: Lowpassed signal output
        """
        self._state = (x * self.gamma) + (1.0 - self.gamma) * self._state
        return self._state

    @property
    def state(self):
        return self._state


class LowpassFIR(object):
    """
   Finite impulse response AKA moving average. Smooths values.
    """

    def __init__(self, init_state, gamma=0.1, size=10):
        """
        :param gamma: Coefficient for lowpass, (0,1]
        gam=1 -> 100% pass
        """
        self.gamma = gamma
        self.size = size
        self._values = []
        self._state = init_state

    def update(self, x):
        """
        Push a value into the filter
        :param x: Value of input signal
        :return: Lowpassed signal output
        """
        self._values.append(x)
        self._values = self._values[-self.size:]
        self._state = sum(self._values) / len(self._values)
        return self._state

    @property
    def state(self):
        return self._state