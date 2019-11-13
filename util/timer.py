import time
from collections import OrderedDict, defaultdict
import pprint


class Timer(object):
    """
    Timer.
    """
    def __init__(self, time_target=0, left_shift=0):
        """
        Initialize Timer.
        :param time_target: target time in seconds
        :type time_target: float
        """
        self._start = time.time()
        self.leftshift(left_shift)
        self._time_target = time_target
        self.log = OrderedDict()
        self.counter = defaultdict(int)

    def restart(self):
        """
        Restart timer.
        """
        self._start = time.time()

    def elapsed(self):
        """
        Get elapsed time since start as secs.
        :return: elapsed time.
        :rtype: float
        """
        return time.time() - self._start

    def finished(self):
        """
        Returns whether or not desired time duration has passed.
        :return: whether or not desired time duration has passed.
        :rtype: bool
        """
        return self.elapsed() >= self._time_target

    @staticmethod
    def sleep(duration):
        """
        Sleep for a while.
        :param duration: duration to sleep
        :type duration: float
        """
        time.sleep(duration)

    def leftshift(self, duration):
        """
        Leftshift timer.
        :param duration: duration to leftshift
        :type duration: float
        """
        self._start -= duration

    def log_and_restart(self, text: str):
        # print(f'[{text}] {self.elapsed()}')
        if text not in self.log:
            self.log[text] = 0
        self.log[text] += self.elapsed()
        self.counter[text] += 1
        self.restart()

    def reset_log(self):
        self.log = OrderedDict()
        self.counter = OrderedDict()

    def print_log(self):
        print()
        print('Profiling')
        print('-' * 30)
        print('{:<20} {:<10}'.format('Item', 'Time (ms)'))
        print('-' * 30)
        for k, v in self.log.items():
            print('{:<20} {:<10.4f}'.format(k, v * 1000 / self.counter[k]))
        print('-' * 30)
