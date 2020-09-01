import numpy as np
from game.cursors import Cursor2D

class Obsercation2D:
    def to_dict(self):
        return self.__dict__

class EnvObservation2D:
    def __init__(self, source_oservation, result_observation):
        """

        :param source_oservation: of at least 2D
        :param result_observation:  of at least 2D
        """
        self.source_observation = source_oservation
        self.result_observation = result_observation

class ModelObservation2D:
    def __init__(self, source_data, result_data):
        """

        :param source_data: flat
        :param result_data:  flat
        """
        assert len(source_data.shape) >= 2
        assert source_data.shape[0] == 1
        assert len(result_data.shape) >= 2
        assert result_data.shape[0] == 1
        self.source_data = source_data
        self.result_data = result_data

    def as_array(self):
        return np.concatenate([self.source_data, self.result_data])

    def as_tuple(self):
        return self.source_data, self.result_data

class Observation2DSettings:
    def __init__(self, cursor: Cursor2D, categories=0):
        self.cursor = cursor
        self.categories = categories
        self.result_shape = self.cursor.region_result_input.shape
        if categories != 0:
            self.result_shape = self.cursor.region_result_input.shape + (self.categories, )

    def game_to_model_observation(self, obs: EnvObservation2D) -> ModelObservation2D:
        ob_src = obs.source_observation.flatten()
        ob_res = obs.result_observation.flatten()
        ob_src, ob_res = ob_src[np.newaxis,:], ob_res[np.newaxis,:]
        return ModelObservation2D(ob_src, ob_res)

    def model_to_game_observation(self, obs: ModelObservation2D) -> EnvObservation2D:
        ob_src = obs.source_data.reshape(self.cursor.region_source_input.shape)
        ob_res = obs.result_data.reshape(self.cursor.region_result_input.shape)
        return EnvObservation2D(ob_src, ob_res)

    def to_network_input(self, obs: ModelObservation2D) -> np.ndarray:
        return [obs.source_data[np.newaxis,:], obs.result_data[np.newaxis,:]]

