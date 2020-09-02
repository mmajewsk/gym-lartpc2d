import math

import numpy as np
from lartpc_game.game.dims import neighborhood2d, neighborhood3d, set_range2d, set_range3d, get_range2d, get_range3d
from abc import abstractmethod

class BaseRegion:
    def __init__(self, size):
        self.range = self.make_range(size)
        self.neighbourhood = self.range[~np.all(self.range == 0, axis=1)]
        self.window_size = size
        self.shape = (size, size)
        self.basic_clock_number = None
        self.r_low = math.floor(self.window_size / 2)
        self.r_high = math.ceil(self.window_size / 2)

    @abstractmethod
    def make_range(self, size):
        return None
    @staticmethod
    @abstractmethod
    def get_range(arr, center_xyz, low, high):
        pass

    @staticmethod
    @abstractmethod
    def set_range(arr, value, center_xyz, low, high):
        pass

class Region2D(BaseRegion):
    def __init__(self, size=2):
        BaseRegion.__init__(self, size)
        self.basic_block_size = self.window_size ** 2

    def make_range(self, size):
        return  neighborhood2d(size)

    @staticmethod
    def get_range(arr, center_xyz, low, high):
        return get_range2d(arr, center_xyz, low, high)

    @staticmethod
    def set_range(arr, value, center_xyz, low, high):
        return set_range2d(arr, value, center_xyz, low, high)

class Region3D(BaseRegion):
    def __init__(self, size=3):
        BaseRegion.__init__(self, size)
        self.range = neighborhood3d(size)

    @staticmethod
    def get_range(arr, center_xyz, low, high):
        return get_range3d(arr, center_xyz, low, high)

    @staticmethod
    def set_range(arr, value, center_xyz, low, high):
        return set_range3d(arr, value, center_xyz, low, high)
