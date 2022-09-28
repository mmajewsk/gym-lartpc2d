import gym
import numpy as np

from lartpc_game.agents.observables import Action2Dai, State2Dai
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.data import LartpcData
import os



def _reward_calc(game, source_cursor, canvas_cursor, target_cursor):
    nonzero_source_px = np.count_nonzero(source_cursor)
    if len(canvas_cursor.shape) == 3:
        canvas_categorised = np.argmax(canvas_cursor, axis=2)
    else:
        canvas_categorised = canvas_cursor
    center_pixel = source_cursor[
        game.cursor.region_source_input.r_low, game.cursor.region_source_input.r_low
    ]
    discovered_pixels = (
        game.cursor.region_canvas_input.basic_block_size
        - np.count_nonzero(canvas_categorised )
    )
    reward = nonzero_source_px + discovered_pixels * 0.09
    assert reward >= 0
    if discovered_pixels == 0:
        reward = reward - 5
    center_pixel_is_zero = np.count_nonzero(center_pixel) == 0
    if center_pixel_is_zero:
        reward = reward - 15
        # if self.reward_history <= 0.0:
        #    reward = reward - self.reward_history[-1]*3
    return reward


def _reward_calc2(game, source_cursor, canvas_cursor, target_cursor):
    times_pixel_touched = game.cursor_history_counter[tuple(game.cursor.current_center)]
    f = lambda x : (((x-1)**1.75)*0.25).real
    revisit_punishment = f(times_pixel_touched)
    cls_multiplier = [1.     , 3.98, 6.79]
    prediction = canvas_cursor[1,1] # e.g [0.3, 0.8, 0.4]
    prediction_class = prediction.argmax() # 2
    target_class = target_cursor[0,0] # e.g 1
    # breakpoint()
    if prediction_class == target_class:
        guess_reward = prediction[target_class]*cls_multiplier[target_class]
    else:
        guess_reward = -1*prediction[prediction_class]*cls_multiplier[prediction_class]
    reward = guess_reward - revisit_punishment
    return reward

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
        self.reward_func = _reward_calc2

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
