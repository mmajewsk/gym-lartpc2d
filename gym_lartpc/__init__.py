from gym.envs.registration import register
import lartpc_game.agents.agents
import lartpc_game.agents.observables
import lartpc_game.agents.settings
import lartpc_game.agents.tools
import lartpc_game.agents
import lartpc_game.game
import lartpc_game.data
import lartpc_game.viz
from lartpc_game import *
from lartpc_game.agents import *
__all__ = ['agents','game','data','viz','agents.agents']

register(
    id='lartpc-v0',
    entry_point='gym_lartpc.envs.lartpc:LartpcEnv',
)
