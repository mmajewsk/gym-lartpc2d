import gym
from gym import error, spaces, utils
from gym.utils import seeding
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.agents.observables import Action2Dai, State2Dai, Observation2Dai

class LartpcEnv(gym.Env, Lartpc2D):
  metadata = {'render.modes': ['human']}

  def __init__(self, result_dimension=3, max_step_number=8):
      gym.Env.__init__(self)
      Lartpc2D.__init__(self, result_dimension, max_step_number)

  def step(self, action: Action2Dai) -> State2Dai:
      return Lartpc2D.step(self, action)

  def set_maps(self, src, trgt):
      self.detector.set_maps(src, trgt)

  def reset(self):
      src, trgt = self.detector.source_map, self.detector.target_map
      self.detector.set_maps(src, trg)
      Lartpc2D.start(self)

  def render(self, mode='human'):
      raise NotImplementedError('Use vis.py insted')

  def close(self):
      pass
