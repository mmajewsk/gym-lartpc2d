from gym.envs.registration import register

register(
    id='lartpc-v0',
    entry_point='gym_lartpc.envs.lartpc:LartpcEnv',
)
