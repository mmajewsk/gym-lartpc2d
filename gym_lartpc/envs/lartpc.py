import gym
import numpy as np

from lartpc_game.agents.observables import Action2Dai, State2Dai
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.data import LartpcData
import os


class LartpcEnv(gym.Env, Lartpc2D):
    metadata = {"render.modes": ["human"]}

    def __init__(self, result_dimension=3, max_step_number=8, max_trials=32):
        gym.Env.__init__(self)
        Lartpc2D.__init__(self, result_dimension, max_step_number)
        self.action_space = gym.spaces.Box(0, 1, (result_dimension + 8,), np.float64)
        self.max_trials = max_trials
        self._trials = 0
        MAX_SOURCE_VAL = 25000
        data_path = os.environ["DATA_PATH"]
        self.data_generator = LartpcData.from_path(data_path)
        source_size = (self.cursor.input_source_size, self.cursor.input_source_size)
        canvas_size = (self.cursor.input_canvas_size, self.cursor.input_canvas_size, 3)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "source": gym.spaces.Box(
                    0, MAX_SOURCE_VAL, source_size, dtype=np.float64
                ),
                "canvas": gym.spaces.Box(0, 1, canvas_size, dtype=np.float64),
            }
        )

    def step(self, action: np.ndarray) -> State2Dai:
        mov_1hot, put_prob = action[:-3], action[-3:]
        mov_i = np.argmax(mov_1hot)
        if mov_i >= 4:
            mov = mov_i + 1
        else:
            mov = mov_i
        mov_x = mov % 3 - 1
        mov_y = mov // 3 - 1
        movement_vector = np.array((mov_x, mov_y)).reshape((1,2))
        put_data = put_prob.reshape((1, 1, 3))
        action = Action2Dai(movement_vector, put_data)
        # breakpoint()
        state = Lartpc2D.step(self, action)
        self._trials += 1
        if self._trials >= self.max_trials:
            self.reset()
            self._trials = 0
        obs = {"source": state.obs.source, "canvas": state.obs.result}
        rval = obs, state.reward, state.done, state.info
        return rval

    def set_maps(self, src, trgt):
        self.detector.set_maps(src, trgt)

    def reset(self):
        map_number = np.random.randint(1000, len(self.data_generator))
        self.detector.set_maps(*self.data_generator[map_number])
        self._trials = 0
        Lartpc2D.start(self)
        state = self.get_state()
        obs = {"source": state.obs.source, "canvas": state.obs.result}
        rval = obs
        return rval

    def render(self, mode="human"):
        raise NotImplementedError("Use vis.py insted")

    def close(self):
        pass
