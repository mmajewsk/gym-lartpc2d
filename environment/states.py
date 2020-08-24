import numpy as np
from environment.observations import GameObservation2D, ModelObservation2D, Observation2DFactory
from tensorflow.keras.utils import to_categorical

class State2D:
    def to_dict(self):
        d = self.__dict__
        d['obs'] = d['obs'].to_dict()
        return d


class GameVisibleState2D(State2D):
    """
    This is a class that encapsulates the state of the game visible
    currently for the actor, produced by Game.
    """
    def __init__(self, obs: GameObservation2D, target, reward, done, info):
        self.obs = obs
        self.target = target
        self.reward = reward
        self.done = done
        self.info = info

class ModelVisibleState2D(State2D):
    """
    This is a
    class that encapsulates the state of the game visible currently
    for the actor, in a form interpretable for model.
    """
    def __init__(self, obs:ModelObservation2D, target, reward, done, info):
        self.obs = obs
        assert len(target.shape) >= 2
        assert target.shape[0] == 1
        self.target = target
        self.reward = reward
        self.done = done
        self.info = info

class VisibleState2DFactory:

    def __init__(self, obsf:Observation2DFactory):
        self.obsf = obsf
        self.cursor = obsf.cursor.copy()
        self.categories = obsf.categories
        self.target_shape = (1,1, obsf.categories)

    def game_to_model_visible_state(self, state: GameVisibleState2D) -> ModelVisibleState2D:
        assert isinstance(state, GameVisibleState2D)
        target_categorical = to_categorical(state.target, num_classes=self.categories)[np.newaxis, :]
        state = ModelVisibleState2D(
            obs= self.obsf.game_to_model_observation(state.obs),
            target=target_categorical,
            reward=state.reward,
            done=state.done,
            info=state.info
        )
        return state

    def model_to_game_visible_state(self, state: ModelVisibleState2D) -> GameVisibleState2D:
        assert isinstance(state, ModelVisibleState2D)
        state = GameVisibleState2D(
            obs = self.obsf.model_to_game_observation(state.obs),
            target=np.argmax(state.target, state.target.shape[-1])
        )
        return state


