from gym.envs.registration import register

register(
    id='Foraging-v0',
    entry_point='Foraging_v0.envs:ForagingEnv',
    kwargs={'num_agents': 1,
            'num_obst': 4,
            'target': False,
            'map_size': False,
            'init_pos': False
            }
    )
