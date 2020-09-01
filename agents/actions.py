import numpy as np

from game.cursors import Cursor2D


class EnvAction2D:
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

class AgentAction2D:
    def __init__(self, put_decision):
        self.put_decision = put_decision

    def _get_simple_index(self, factory) -> int:
        pass

    def to_game_aciton(self, factory) -> EnvAction2D:
        # this is loss convertion
        # for window_size =3
        # middle index = floor(3/2)*(3+1)
        # this calculates central pixel
        middle_index = factory.cursor.region_movement.r_low * (factory.cursor.region_movement.window_size + 1)
        simple_index = self._get_simple_index(factory)
        if simple_index > middle_index:
            a = simple_index + 1
        else:
            a = simple_index + 0
        unflat_data = self.put_decision.reshape(factory.put_shape)
        return EnvAction2D(factory.possible_movement[simple_index], simple_index, unflat_data)

class QAction2D(AgentAction2D):

    def __init__(self, movement_decision, put_decision, factory=None):
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
        AgentAction2D.__init__(self, put_decision)
        self.movement_decision= movement_decision
        self.factory = factory

    def _get_simple_index(self, factory):
        return np.argmax(self.movement_decision)

    @staticmethod
    def create_random_action(factory):
        movement_random = np.random.random(factory.movement_size)
        put_random = np.random.random(factory.put_shape)
        return QAction2D(movement_random, put_random)

    def randomise_movement(self, factory):
        assert self.put_decision.shape==(1, factory.data_size), "Incorrect shape, expected {} got {}".format((1,factory.movement_size), self.put_decision.shape)
        dummy_model = self.create_random_action()
        self.movement_decision = dummy_model.movement_decision
        return dummy_model

    def from_game(self, g_a: EnvAction2D, factory):
        # this is loss convertion
        flat_movement = np.zeros(factory.movement_size, factory.possible_movement.dtype)
        a, = np.where(factory.possible_movement==g_a.movement_vector)
        flat_movement[a] = 1.0
        return QAction2D(flat_movement.flatten(), g_a.put_data)

    def randomise_category(self, factory) :
        assert self.movement_decision.shape==(1,1,factory.movement_size), "Incorrect shape, expected {} got {}".format((1, factory.movement_size), self.movement_decision.shape)
        dummy_model = self.create_random_action()
        self.put_decision = dummy_model.put_decision
        return dummy_model

    @staticmethod
    def from_flat(flat_array: np.ndarray, factory) :
        assert len(flat_array)== factory.data_size+factory.movement_size, "Incorrect array length"
        flat_movement, flat_data = flat_array[:factory.movement_size], flat_array[factory.movement_size:]
        return QAction2D(flat_movement, flat_data)

    def to_flat(self) -> np.ndarray:
        return np.concatenate([self.movement_decision.flatten(), self.put_decision.flatten()])

    def __str__(self):
        return "move: {} \n put: {}".format(self.movement_decision, self.put_decision)

class PolicyAction(AgentAction2D):
    def __init__(self, policy, put_decision):
        self.policy = policy.flatten()
        AgentAction2D.__init__(self,put_decision)

    def _get_simple_index(self, factory):
        return np.random.choice(factory.movement_size, 1, p=self.policy)[0]

    @staticmethod
    def from_game(g_a: EnvAction2D, factory):
        # this is loss convertion
        flat_movement = np.zeros(factory.movement_size, factory.possible_movement.dtype)
        a = np.where((factory.possible_movement==g_a.movement_vector).all(axis=1))
        flat_movement[a] = 1.0
        return PolicyAction(flat_movement.flatten(), g_a.put_data)


    def to_game_aciton(self, factory) -> EnvAction2D:
        return AgentAction2D.to_game_aciton(self, factory)

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





