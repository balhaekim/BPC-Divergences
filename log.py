import time
import logging
from absl import logging as absl_logging
from collections import OrderedDict

def get_logger(filename, mode='a'):

    # due to the inteference between absl logging and python logging,
    # manually set absl logging format
    formatter = logging.Formatter('[%(asctime)s]: %(message)s', '%Y/%m/%d-%H:%M')
    absl_logging.get_absl_handler().setFormatter(formatter)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    return logger

class RunningAverage(object):
    def __init__(self, *keys):
        self.dict = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.dict[key] = (0, 0)

    def update(self, key, val):
        if self.dict.get(key, None) is None:
            self.dict[key] = (val, 1)
        else:
            self.dict[key] = (self.dict[key][0]+val, self.dict[key][1]+1)

    def reset(self):
        for key in self.dict.keys():
            self.dict[key] = (0, 0)
        self.clock = time.time()

    def clear(self):
        self.dict = OrderedDict()
        self.clock = time.time()

    def keys(self):
        return self.dict.keys()

    def get(self, key):
        entry = self.dict.get(key, None)
        assert(entry is not None)
        return entry[0] / entry[1]

    def info(self, show_et=True):
        line = ''
        for key, (val, cnt) in self.dict.items():
            if cnt > 0:
                line += f'{key} {val/cnt:.4f} '
        if show_et:
            line += f'({time.time()-self.clock:.3f} secs)'
        return line
