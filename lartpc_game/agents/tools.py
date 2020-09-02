import numpy as np
from lartpc_game.agents.settings import Action2DSettings
from lartpc_game.agents.observables import Action2Dai

def to_categorical_(y, num_classes=None, dtype='float32'):
    # taken from https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class PolicyToAction:
    def __call__(self, output, action_settings: Action2DSettings):
        mov, put = output
        simple_index = self._get_simple_index(mov)
        unflat_data = put.reshape(action_settings.put_shape)
        new_mov = action_settings.possible_movement[simple_index]
        new_mov = new_mov[np.newaxis,:]
        return Action2Dai(new_mov, unflat_data)

    def _get_simple_index(self, mov):
        return np.argmax(mov)
