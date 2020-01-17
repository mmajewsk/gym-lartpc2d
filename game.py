from actors.actions import GameAction2D
from actors.observations import GameObservation2D
from actors.states import GameVisibleState2D
from rl_environments.cursors import Cursor2D
import numpy as np
import pandas as pd
from abc import abstractmethod, ABC

class BaseEnvironment(ABC):
    def __init__(self, result_dimensions = None):
        self.source_map = None
        self.target_map = None
        self.result_map = None
        self.result_dimensions = result_dimensions

    @property
    @abstractmethod
    def dimension_list(self):
        #eg. return ['x','y','z']
        pass

    def read_source_nonzero_indeces(self):
        self.nonzero_indeces = np.where(self.source_map > 0.)

    def create_nonzero_df(self, nonzero_indeces):
        df_dict = {}
        for dim, nonzero_ax in zip(self.dimension_list, nonzero_indeces):
            df_dict[dim] = nonzero_ax
        self.nonzero_df = pd.DataFrame(df_dict)
        self.nonzero_df['touched'] = 0

    def get_random_positions(self):
        row = self.nonzero_df.sample(1)
        return row[self.dimension_list].values

    def set_map(self, source, target, result=None):
        self.source_map = source
        self.target_map = target
        if result is not None:
            self.result_map = result
        else:
            if self.result_dimensions is not None:
                self.result_map = np.zeros(self.target_map.shape+(self.result_dimensions,))
            else:
                self.result_map = np.zeros_like(self.target_map)

        self.read_source_nonzero_indeces()
        self.create_nonzero_df(self.nonzero_indeces)

    def get_maps(self):
        return self.source_map, self.target_map, self.result_map


class Environment2D(BaseEnvironment):

    @property
    def dimension_list(self):
        return ['x','y']


class Game2D:

    def __init__(self, env: Environment2D, max_step_number):
        self.env = env
        self.cursor = Cursor2D(output_size=1, input_result_size = 5, input_source_size=5, movement_size=3)
        self.cursor_history = []
        self.reward_history = []
        self.max_step_number = max_step_number
        self.step_number = None
        self.done = None


    def move_cursor(self, new_center):
        self.cursor_history.append(self.cursor.current_center.copy())
        self.cursor.current_center = new_center

    def start(self):
        self.cursor_history = []
        for center in self.env.get_random_positions():
            if not self._outside_marigin(center):
                self.cursor.current_center = center
                break
        self.step_number = 0
        self.done = False

    def _outside_marigin(self, new_center):
        marigin = self.cursor.region_source_input.r_low
        return np.any(new_center <= marigin) or np.any(new_center >= self.env.source_map.shape[0]-marigin)


    def _act(self, action: GameAction2D) -> bool:
        assert action.put_data.shape==self.cursor.region_output.shape+(3,)
        self.cursor.set_range(self.env.result_map, action.put_data)
        new_center = self.cursor.current_center + action.movement_vector
        if self._outside_marigin(new_center):
            action_success = False
        else:
            self.move_cursor(new_center)
            action_success = action.movement_vector.any()
        return action_success

    def step(self, action: GameAction2D) -> GameVisibleState2D:
        done = True
        can_move = self.step_number < self.max_step_number -1
        last_move = self.step_number == self.max_step_number -1
        if can_move:
            action_success = self._act(action)
            # if action succedded, then it is not done
            if not last_move and action_success:
                done = False
                self.step_number += 1
            else:
                done = True
        self.done = done
        #if np.count_nonzero(self.get_observation().source_observation) == 0:
        #    done = True
        state = self.get_state()
        self.reward_history.append(state.reward)
        return state

    def get_observation(self) -> GameObservation2D:
        source_curs = self.cursor.get_range(self.env.source_map, region_type='source_input')
        result_curs = self.cursor.get_range(self.env.result_map, region_type='result_input')
        obs = GameObservation2D(source_curs, result_curs)
        return obs

    def get_target(self) -> np.ndarray:
        target_curs = self.cursor.get_range(self.env.target_map, region_type='output')
        return target_curs

    def get_state(self):
        return GameVisibleState2D(self.get_observation(), self.get_target(), self.reward(), self.done, "")

    @staticmethod
    def _reward_calc(game, source_cursor, result_cursor):
        nonzero_source_px = np.count_nonzero(source_cursor)
        if len(result_cursor.shape) == 3:
            result_categorised = np.argmax(result_cursor, axis=2)
        else:
            result_categorised = result_cursor
        center_pixel = source_cursor[game.cursor.region_source_input.r_low, game.cursor.region_source_input.r_low]
        discovered_pixels = game.cursor.region_result_input.basic_block_size - np.count_nonzero(result_categorised)
        reward = nonzero_source_px+discovered_pixels*.09
        assert reward>=0
        if discovered_pixels== 0:
            reward = reward - 5
        center_pixel_is_zero =  (np.count_nonzero(center_pixel) == 0 )
        if center_pixel_is_zero:
            reward = reward-15
            #if self.reward_history <= 0.0:
            #    reward = reward - self.reward_history[-1]*3
        return reward

    def reward(self):
        rewards_dict = dict(
            source_cursor=self.cursor.get_range(self.env.source_map),
            result_cursor=self.cursor.get_range(self.env.result_map, region_type='result_input'),
        )
        reward = Game2D._reward_calc(self, **rewards_dict)
        return reward


