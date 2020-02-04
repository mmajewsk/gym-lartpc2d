import numpy as np

from game.cursors import Cursor2D


class GameAction2D:
    def __init__(self, movement_vector, movement_number, put_data):
        """
        This is accurate action that is supposed to be taken
        :param movement_vector: e.g [-1,0]
        :param put_data: [
        [[0, 0, 0, 1],[...],[...]],
        [[...],[...],[...]]
        ....

        ]
        """
        self.movement_number = movement_number
        self.movement_vector = movement_vector
        self.put_data = put_data

    def __str__(self):
        return "={}=\n ={}= \n ={}=".format(self.movement_number, self.movement_vector, self.put_data)

class ModelAction2D:
    def __init__(self, movement_decision, put_decision):
        """
        This action means the output from the model it is expressed as probabilities.
        :param movement_decision: e.g: [0.3, 0.55, 0.98, ...] - no one apparent
        :param put_decision: basically np.random.rand((3,3,4)) or other shape
        e.g.:

        array([
            [[0.20231526, 0.43392942, 0.34223859, 0.51505905],
            [0.21615184, 0.52743003, 0.38137611, 0.89218293],
            [0.04797466, 0.85458022, 0.6446696 , 0.35271566]],

           [[0.13952278, 0.24014195, 0.71320823, 0.17704216],
            [0.11197159, 0.74247326, 0.14831169, 0.67427213],
            [0.72906299, 0.21617329, 0.25758195, 0.40932484]],

           [[0.39746772, 0.37090612, 0.35084151, 0.0370317 ],
            [0.64722792, 0.45966011, 0.53903709, 0.39774236],
            [0.86995326, 0.28019281, 0.69174254, 0.96382261]]
        ])

        """
        self.movement_decision= movement_decision
        self.put_decision = put_decision

    def __str__(self):
        return "move: {} \n put: {}".format(self.movement_decision, self.put_decision)

class Action2DFactory:
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

    def game_action_to_model(self, g_a: GameAction2D) -> ModelAction2D:
        # this is loss convertion
        flat_movement = np.zeros(self.movement_size, self.possible_movement.dtype)
        a, = np.where(self.possible_movement==g_a.movement_vector)
        flat_movement[a] = 1.0
        return ModelAction2D(flat_movement.flatten(), g_a.put_data)

    def model_action_to_game(self, m_a: ModelAction2D) -> GameAction2D:
        # this is loss convertion
        # for window_size =3
        # middle index = floor(3/2)*(3+1)
        # this calculates central pixel
        middle_index = self.cursor.region_movement.r_low*(self.cursor.region_movement.window_size+1)
        simple_index = np.argmax(m_a.movement_decision)
        if simple_index>middle_index:
            a = simple_index + 1
        else:
            a = simple_index + 0
        unflat_data = m_a.put_decision.reshape(self.put_shape)
        return GameAction2D(self.possible_movement[simple_index], simple_index, unflat_data)

    def flat_to_model_action(self, flat_array: np.ndarray) -> ModelAction2D:
        assert len(flat_array)==self.data_size+self.movement_size, "Incorrect array length"
        flat_movement, flat_data = flat_array[:self.movement_size], flat_array[self.movement_size:]
        return ModelAction2D(flat_movement, flat_data)

    def randomise_category(self, action: ModelAction2D) -> ModelAction2D:
        assert action.movement_decision.shape==(1,1,self.movement_size), "Incorrect shape, expected {} got {}".format((1,self.movement_size), action.movement_decision.shape)
        dummy_model = self.create_random_action()
        action.put_decision = dummy_model.put_decision
        return dummy_model

    def randomise_movement(self, action: ModelAction2D) -> ModelAction2D:
        assert action.put_decision.shape==(1,self.data_size), "Incorrect shape, expected {} got {}".format((1,self.movement_size), movement.shape)
        dummy_model = self.create_random_action()
        action.movement_decision = dummy_model.movement_decision
        return dummy_model

    def model_action_to_flat(self, action: ModelAction2D) -> np.ndarray:
        return np.concatenate([action.movement_decision.flatten(), action.put_decision.flatten()])

    def create_random_action(self) -> ModelAction2D:
        movement_random = np.random.random(self.movement_size)
        put_random = np.random.random(self.put_shape)
        return ModelAction2D(movement_random, put_random)

