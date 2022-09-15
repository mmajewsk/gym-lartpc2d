import gym
import numpy as np

from lartpc_game.agents.observables import Action2Dai, State2Dai
from lartpc_game.game.game_ai import Lartpc2D


class LartpcEnv(gym.Env, Lartpc2D):
    metadata = {"render.modes": ["human"]}

    def __init__(self, result_dimension=3, max_step_number=8):
        gym.Env.__init__(self)
        Lartpc2D.__init__(self, result_dimension, max_step_number)
        self.action_space = gym.spaces.Box(0, 1, (result_dimension+8,), np.float64)
        MAX_SOURCE_VAL = 25000

        source_size = (self.cursor.input_source_size, self.cursor.input_source_size)
        canvas_size = (self.cursor.input_canvas_size, self.cursor.input_canvas_size, 3)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "source": gym.spaces.Box(0, MAX_SOURCE_VAL, source_size, dtype=np.float64),
                "canvas": gym.spaces.Box(0, 1, canvas_size, dtype=np.float64),
            }
        )

    def step(self, action: np.ndarray) -> State2Dai:
        mov_1hot, put_prob = action[:-3], action[-3:]
        mov_i = np.argmax(mov_1hot)
        if mov_i>=4:
          mov = mov_i+1
        else:
          mov = mov_i
        mov_x = mov%3-1
        mov_y = mov//3-1
        movement_vector = np.array((mov_x,mov_y))
        put_data = put_prob.reshape((1,1,3))
        action = Action2Dai(movement_vector, put_data)
        return Lartpc2D.step(self, action)

    def set_maps(self, src, trgt):
        self.detector.set_maps(src, trgt)

    def reset(self):
        src, trgt = self.detector.source_map, self.detector.target_map
        self.detector.set_maps(src, trgt)
        Lartpc2D.start(self)

    def render(self, mode="human"):
        raise NotImplementedError("Use vis.py insted")

    def close(self):
        pass
