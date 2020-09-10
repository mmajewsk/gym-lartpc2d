import random
import numpy as np
from lartpc_game.game.game_ai import Lartpc2D
from lartpc_game.agents.settings import Action2DSettings, Observation2DSettings
from lartpc_game.agents.observables import State2Dai


class BaseAgent:
    def __init__(
            self,
            env: Lartpc2D
         ):
        self.env = env
        self.action_settings =  self.env.action_settings
        self.observation_settings = self.env.observation_settings


    def create_action(self, state: State2Dai):
        pass


