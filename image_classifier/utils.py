import os
from functools import lru_cache

import torch


class ValueCache(object):
    def __init__(self):
        self.reset()

    def update(self, value, count=1):
        self.value = value
        self.count += count
        self.sum += value
        self.mean = self.sum / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0.0
        self.sum = 0.0
        self.mean = 0.0


# System Information Helpers
@lru_cache(maxsize=2)
def get_device(allow_gpu):
    if allow_gpu & torch.cuda.is_available():
        print('Using GPU')
        return torch.device('cuda')
    else:
        print('Using CPU')
        return torch.device('cpu')


@lru_cache(maxsize=1)
def get_max_workers():
    return os.cpu_count()
