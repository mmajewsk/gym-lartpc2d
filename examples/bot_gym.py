import numpy as np
import argparse
from lartpc_game import data
from lartpc_game.agents.agents import BaseAgent
from lartpc_game.agents.observables import Observation2Dai, State2Dai, Action2Dai
from lartpc_game.game.dims import neighborhood2d
from lartpc_game.agents.tools import PolicyToAction
import gym
import gym_lartpc



class BotAgent(BaseAgent):
    def __init__(
            self,
            env
        ):
        BaseAgent.__init__(self, env)
        input_size = self.observation_settings.cursor.region_source_input.window_size
        assert input_size==5
        large_neighborhood = neighborhood2d(input_size)
        small_neighborhood = neighborhood2d(input_size-2)
        large_neighborhood= large_neighborhood[~np.all(large_neighborhood == 0, axis=1)]
        small_neighborhood= small_neighborhood[~np.all(small_neighborhood == 0, axis=1)]

        self.lu_small_nbhood = 2 + small_neighborhood
        self.lu_large_nbhood = 2 + large_neighborhood
        empty = np.zeros((input_size,input_size))
        smlcl = self.lu_small_nbhood.T
        empty[smlcl[0], smlcl[1]] = True
        self.lu_small_nbhood_mask = empty.astype(np.bool)
        empty2 = np.zeros((input_size,input_size))
        lrgcl = self.lu_large_nbhood.T
        empty2[lrgcl[0], lrgcl[1]] = True
        empty2[self.lu_small_nbhood_mask] = False
        self.lu_large_nbhood_mask = empty2.astype(np.bool)


    def create_action(self, state: State2Dai) -> Action2Dai:
        """
         Ok so:
         1. check nearest neighbours, if have value, and untoched, move at random. If not possible:
         2. check outer ring, the same procedure
        :param state:
        :return:
        """

        smbhd_source = state.source[self.lu_small_nbhood_mask]
        smbhd_result = state.result[self.lu_small_nbhood_mask]
        assert smbhd_result.shape[-1] == 3
        smbhd_result = np.argmax(smbhd_result, axis=1)
        go = (smbhd_result == 0) & (smbhd_source != 0)
        go_indeces = np.nonzero(go)[0]
        if go_indeces.size == 0:
            desperate_move = np.nonzero(smbhd_source)[0]
            if desperate_move.size==0:
                go_indeces = np.array(range(8))
            else:
                go_indeces = desperate_move
        choice = np.random.choice(go_indeces,1)[0]
        result = np.zeros((1,8))
        result[0,choice] = 1.0
        movement_random = np.random.random(self.action_settings.movement_size).astype(np.float32)
        put_random = np.random.random(self.action_settings.put_shape).astype(np.float32)
        action = PolicyToAction()((movement_random, put_random), self.action_settings)
        action.type_check()
        return action

def bot_replay(data_path, viz=True):
    max_step_number = 20
    data_generator = data.LartpcData.from_path(data_path)
    result_dimension = 3
    kwargs = dict(result_dimension=result_dimension, max_step_number=max_step_number)
    game = gym.make("lartpc-v0", **kwargs)
    if viz:
        # i know this is not nice, but sometimes opencv can be stack at debug
        from lartpc_game.viz import Visualisation
        vis = Visualisation(game)
    agent = BotAgent(game)
    for iterate_maps in range(30):
        map_number = np.random.randint(0, len(data_generator))
        game.set_maps(*data_generator[map_number])
        for iterate_tries in range(10):
            game.start()
            for model_run_iteration in range(game.max_step_number):
                current_observation = game.get_observation()
                action = agent.create_action(current_observation)
                state = game.step(action)
                if viz: vis.update(0)
                if state.done:
                    break



if __name__ == "__main__":
    description = """
    Runs a simple bot showcasing the game.

    e.g. usage
        bot.py ../../assets/dump\n
        bot.py ../../assets/dump --viz-off\n
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', help='Path to the data generated from lartpc')
    parser.add_argument('--viz-off', default=True, action='store_false', help='Run without visualisation/opencv (helpful for debug)')
    args = parser.parse_args()
    bot_replay(args.data_path, viz=args.viz_off)
