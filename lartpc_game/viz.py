import cv2
from collections import OrderedDict
import numpy as np
from lartpc_game.game import game_ai
import matplotlib.pyplot as plt
from lartpc_game.agents.observables import Action2Dai, State2Dai, Observation2Dai

class VisMap:
    def __init__(self, image, cmap_name='viridis'):
        self.img = image
        self.cmapname = cmap_name
        self.cmap = None
        self.norm = None

    def calculate_heatmap(self, img):
        self.cmap = plt.cm.get_cmap(self.cmapname)
        self.norm = plt.Normalize(vmin=img.min(), vmax=img.max())

    def heat_image(self):
        if self.cmap is None or self.norm  is None:
            self.calculate_heatmap(self.img)
        return self.cmap(self.norm(self.img))

    def reverse_heat(self, image):
        return  self.cmap.reverse(self.norm.inverse(image))



class Visualisation:
    def __init__(self, game: game_ai.Lartpc2D):
        self.game = game

    @property
    def heatmaps(self):
        showmaps = [('source', self._heat_source_map),
        ('target', self._heat_target_map),
        ('result', self._heat_result_map)]
        return OrderedDict(showmaps)

    @property
    def window_positions(self):
        win_poses = [
            ('source', (25, 260)),
            ('target', (450, 260)),
            ('result', (900, 260)),
            ('source_cursor', (25, 100)),
            ('target_cursor', (450, 100)),
            ('result_cursor', (900, 100))
        ]
        return OrderedDict(win_poses)

    @heatmaps.setter
    def heatmaps(self, val):
        assert len(val) == 3, "setting too long"
        self._heat_source_map = val[0]
        self._heat_target_map = val[1]
        self._heat_result_map = val[2]

    def _update_maps(self):
        self._source_img = self.game.detector.source_map.copy()
        self._target_img = self.game.detector.target_map.copy()
        _result_img = self.game.detector.result_map.copy()
        self._result_img = np.argmax(_result_img, axis=2)
        self._vis_source_map = VisMap(self._source_img)
        self._vis_target_map = VisMap(self._target_img)
        self._vis_result_map = VisMap(self._result_img)
        self._heat_source_map = self._vis_source_map.heat_image()
        self._heat_target_map = self._vis_target_map.heat_image()
        self._heat_result_map = self._vis_result_map.heat_image()

    def add_cursor_to_maps(self):
        for _, hmap in self.heatmaps.items():
            self.game.cursor.set_range(hmap, np.array([1,0,0, 1.]), region_type='source_input')

    def show_cursor(self, name, data):
        cv2.imshow(name, data)

    def draw_cursor_values(self):
        for name, data in self.heatmaps.items():
            cursor_data = self.game.cursor.get_range(data)
            name = '{}_cursor'.format(name)
            cv2.imshow(name,cursor_data)

    def draw_heatmaps(self):
        for name, map in self.heatmaps.items():
            map = cv2.resize(map, (400,400))
            cv2.imshow(name, map)

    def move_windows(self):
        for name, position in self.window_positions.items():
            cv2.moveWindow(name, *position)

    def draw(self):
        self._update_maps()
        self.draw_cursor_values()
        self.add_cursor_to_maps()
        self.draw_heatmaps()

    def update(self, wait=0):
        self.draw()
        self.move_windows()
        cv2.waitKey(wait)

class MixedModelVisualisation(Visualisation):

    @property
    def window_positions(self):
        poses = super().window_positions
        poses['network_input'] = (1200, 100)
        poses['network_output'] = (1200, 250)
        return poses

    def obs_action(self, obs: Observation2Dai, action: Action2Dai):
        self.observation = obs
        self.action = action

    def draw_network_io(self):
        input = self._vis_source_map.cmap(self._vis_source_map.norm(self.observation.source_observation))
        cv2.imshow('network_input', input)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.bar([0,1,2],self.action.put_data[0][0])
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        X = cv2.resize(X,(150,150))
        cv2.imshow('network_output', X)

    def draw(self):
        super().draw()
        self.draw_network_io()



