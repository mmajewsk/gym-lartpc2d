import numpy as np
from lartpc_game.game.cursors import Cursor2D

class Action2DSettings:
    def __init__(self, cursor: Cursor2D, categories=0):
        self.cursor = cursor
        mov_range = cursor.region_movement.neighbourhood
        """
        for range see  neighborhood2d
        """
        self.possible_movement = mov_range
        self.possible_data = cursor.region_output.range
        self.movement_size = cursor.region_movement.basic_block_size-1
        self.data_size =  cursor.region_output.basic_block_size
        self.categories = categories
        self.put_shape = self.cursor.region_output.shape
        if self.categories!=0:
            self.put_shape = self.put_shape+(self.categories, )

class Observation2DSettings:
    def __init__(self, cursor: Cursor2D, categories=0):
        self.cursor = cursor
        self.categories = categories
        self.result_shape = self.cursor.region_result_input.shape
        if categories != 0:
            self.result_shape = self.cursor.region_result_input.shape + (self.categories, )
