import data
from actors.base_models import BaseActor
from game.game import Environment2D,Game2D
from actors.actions import Action2DFactory, ModelAction2D
from actors.observations import Observation2DFactory, GameObservation2D
from game.dims import neighborhood2d
import numpy as np
import argparse

class BotActor(BaseActor):
    def __init__(
            self,
            action_factory: Action2DFactory,
            observation_factory: Observation2DFactory
        ):
        BaseActor.__init__(self, action_factory, observation_factory)
        input_size = self.observation_factory.cursor.region_source_input.window_size
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


    def create_action(self, state: GameObservation2D) -> ModelAction2D:
        """
         Ok so:
         1. check nearest neighbours, if have value, and untoched, move at random. If not possible:
         2. check outer ring, the same procedure
        :param state:
        :return:
        """

        smbhd_source = state.source_observation[self.lu_small_nbhood_mask]
        smbhd_result = state.result_observation[self.lu_small_nbhood_mask]
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
        action = self.action_factory.create_random_action()
        action.movement_decision = result
        return action

def bot_replay(data_path, viz=True):
    max_step_number = 20
    data_generator = data.LartpcData.from_path(data_path)
    result_dimensions = 3
    env = Environment2D(result_dimensions=result_dimensions)
    env.set_map(*data_generator[3])
    game = Game2D(env, max_step_number=max_step_number)
    if viz:
        # i know this is not nice, but sometimes opencv can be stack at debug
        from viz import Visualisation
        vis = Visualisation(game)
    action_factory = Action2DFactory(game.cursor.copy(), categories=result_dimensions)
    observation_factory = Observation2DFactory(game.cursor.copy(), categories=result_dimensions)
    actor = BotActor(
        action_factory,
        observation_factory,
    )
    for iterate_maps in range(30):
        map_number = np.random.randint(0, len(data_generator))
        game.env.set_map(*data_generator[map_number])
        for iterate_tries in range(10):
            game.start()
            for model_run_iteration in range(game.max_step_number):
                current_observation = game.get_observation()
                model_action = actor.create_action(current_observation)
                game_action = actor.action_factory.model_action_to_game(model_action)
                state = game.step(game_action)
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


