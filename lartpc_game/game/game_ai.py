from lartpc_game.agents.settings import Action2DSettings, Observation2DSettings
from lartpc_game.agents.observables import (
    Action2Dai,
    State2Dai,
    Observation2Dai,
    State2Dai,
)
from lartpc_game.game.cursors import Cursor2D
import numpy as np
import pandas as pd

from abc import abstractmethod, ABC


class DetectorMaps(ABC):
    def __init__(self, result_dimensions=None):
        self.source_map = None
        self.target_map = None
        self.canvas_map = None
        self.result_dimensions = result_dimensions

    @property
    @abstractmethod
    def dimension_list(self):
        # eg. return ['x','y','z']
        pass

    def read_source_nonzero_indeces(self):
        self.nonzero_indeces = np.where(self.source_map > 0.0)

    def create_nonzero_df(self, nonzero_indeces):
        df_dict = {}
        for dim, nonzero_ax in zip(self.dimension_list, nonzero_indeces):
            df_dict[dim] = nonzero_ax
        self.nonzero_df = pd.DataFrame(df_dict)
        self.nonzero_df["touched"] = 0

    def get_random_positions(self):
        row = self.nonzero_df.sample(1)
        return row[self.dimension_list].values

    def set_maps(self, source, target, canvas=None):
        self.source_map = source
        self.target_map = target
        if canvas is not None:
            self.canvas_map = canvas
        else:
            if self.result_dimensions is not None:
                self.canvas_map = np.zeros(
                    self.target_map.shape + (self.result_dimensions,)
                )
            else:
                self.canvas_map = np.zeros_like(self.target_map)

        self.read_source_nonzero_indeces()
        self.create_nonzero_df(self.nonzero_indeces)

    def get_maps(self):
        return self.source_map, self.target_map, self.canvas_map


class Detector2D(DetectorMaps):
    @property
    def dimension_list(self):
        return ["x", "y"]


class Lartpc2D:
    def __init__(self, result_dimension: int, max_step_number: int):
        """
        :param result_dimenstion: the number of dimensions in the result map. I.E. if the map is
                                  256x256 and each pixel can be of three categories then this is
                                  equal to 3
        :param max_step_number: the maximum number of steps to be allowed in the environment
        """
        self.result_dimenstion = result_dimension
        self.detector = Detector2D(result_dimension)
        self.cursor = Cursor2D(
            output_canvas_size=1, input_canvas_size=5, input_source_size=5, movement_size=3
        )
        self.cursor_history = []
        self.reward_history = []
        self.max_step_number = max_step_number
        self.step_number = None
        self.done = None
        self.action_settings = Action2DSettings(
            self.cursor.copy(), categories=self.detector.result_dimensions
        )
        self.observation_settings = Observation2DSettings(
            self.cursor.copy(), categories=self.detector.result_dimensions
        )

    def move_cursor(self, new_center):
        self.cursor_history.append(self.cursor.current_center.copy())
        self.cursor.current_center = new_center

    def start(self):
        self.cursor_history = []
        for center in self.detector.get_random_positions():
            if not self._outside_marigin(center):
                self.cursor.current_center = center
                break
        self.step_number = 0
        self.done = False

    def _outside_marigin(self, new_center):
        marigin = self.cursor.region_source_input.r_low
        return np.any(new_center <= marigin) or np.any(
            new_center >= self.detector.source_map.shape[0] - marigin
        )

    def _act(self, action: Action2Dai) -> bool:
        assert action.put_data.shape == self.cursor.region_canvas_output.shape + (3,)
        assert action.movement_vector.shape == (1, 2)
        self.cursor.set_range(self.detector.canvas_map, action.put_data)
        new_center = self.cursor.current_center + np.squeeze(action.movement_vector)
        if self._outside_marigin(new_center):
            action_success = False
        else:
            self.move_cursor(new_center)
            action_success = action.movement_vector.any()
        return action_success

    def step(self, action: Action2Dai) -> State2Dai:
        done = True
        can_move = self.step_number < self.max_step_number - 1
        last_move = self.step_number == self.max_step_number - 1
        if can_move:
            action_success = self._act(action)
            # if action succedded, then it is not done
            if not last_move and action_success:
                done = False
                self.step_number += 1
            else:
                done = True
        self.done = done
        # if np.count_nonzero(self.get_observation().source) == 0:
        #    done = True
        state = self.get_state()
        self.reward_history.append(state.reward)
        return state

    def get_observation(self) -> Observation2Dai:
        source_curs = self.cursor.get_range(
            self.detector.source_map, region_type="source_input"
        ).astype(np.float32)
        canvas_curs = self.cursor.get_range(
            self.detector.canvas_map, region_type="canvas_input"
        ).astype(np.float32)
        target_curs = self.cursor.get_range(
            self.detector.target_map, region_type="canvas_output"
        ).astype(np.int32)
        obs = Observation2Dai(source_curs, canvas_curs, target_curs)
        return obs

    def get_state(self) -> State2Dai:
        return State2Dai(self.get_observation(), self.reward(), self.done, {})

    @staticmethod
    def _reward_calc(game, source_cursor, canvas_cursor):
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

    def reward(self):
        rewards_dict = dict(
            source_cursor=self.cursor.get_range(self.detector.source_map),
            canvas_cursor=self.cursor.get_range(
                self.detector.canvas_map, region_type="canvas_input"
            ),
        )
        reward = Lartpc2D._reward_calc(self, **rewards_dict)
        return reward
