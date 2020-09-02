from dataclasses import dataclass
from nptyping import NDArray
import typing
import numpy as np

class StaticTyped:
    def type_check(self):
        for k,v in typing.get_type_hints(self.__class__).items():
            message =  "Attribute {} of type {} was expected to of type {}".format
            attr_val = getattr(self, k)
            instance_check = isinstance(attr_val, v)
            if not instance_check and isinstance(attr_val, np.ndarray):
                message = "Attribute {} is numpy array of shape and type [{}, {}] but expected {}".format
                assert instance_check, message(k, attr_val.shape, attr_val.dtype, v)
            assert instance_check, message(k, type(getattr(self,k)), v)

class StaticTypedForced(StaticTyped):

    def __post_init__(self):
        self.type_check()

@dataclass
class Observation2Dai(StaticTyped):
    source: NDArray[(typing.Any, typing.Any), np.float32]
    result: NDArray[(typing.Any, typing.Any, 3), np.float32]
    target: NDArray[(typing.Any, typing.Any), np.int32]

@dataclass
class State2Dai(StaticTyped):
    """
    This is a class that encapsulates the state of the game visible
    currently for the actor, produced by Game.
    """
    obs: Observation2Dai
    reward: float
    done: bool
    info: object

@dataclass
class Action2Dai(StaticTyped):
        """
        This is accurate action that is supposed to be taken
        :param movement_vector: e.g [-1,0]
        :param put_data: [
        [[0, 0, 0, 1],[...],[...]],
        [[...],[...],[...]]
        ....

        ]
        """
        movement_vector: NDArray[(1,2), np.int64]
        put_data: NDArray[(typing.Any, typing.Any, 3), np.float32]
