"""
The purpose of this script is to investigate d4rl envs at a high level
"""

import minari
import gym
import d4rl.gym_mujoco
import numpy as np
import pandas as pd
from minari.dataset import MinariDataset

envs=[
    "halfcheetah-random-v0",
    # "hopper-random-v0",
    # "walker2d-random-v0",
    # "halfcheetah-medium-v0",
    # "hopper-medium-v0",
    # "walker2d-medium-v0",
    # "halfcheetah-expert-v0",
    # "hopper-expert-v0",
    # "walker2d-expert-v0",
    # "halfcheetah-medium-expert-v0",
    # "hopper-medium-expert-v0",
    # "walker2d-medium-expert-v0",
    # "halfcheetah-medium-replay-v0",
    # "hopper-medium-replay-v0",
    # "walker2d-medium-replay-v0",
    # "halfcheetah-random-v2",
    # "hopper-random-v2",
    # "walker2d-random-v2",
    # "halfcheetah-medium-v2",
    # "hopper-medium-v2",
    # "walker2d-medium-v2",
    # "halfcheetah-expert-v2",
    # "hopper-expert-v2",
    # "walker2d-expert-v2",
    # "halfcheetah-medium-expert-v2",
    # "hopper-medium-expert-v2",
    # "walker2d-medium-expert-v2",
    # "halfcheetah-medium-replay-v2",
    # "hopper-medium-replay-v2",
    # "walker2d-medium-replay-v2",
    ]

for env in envs:
    envn = gym.make(env)

    # retrieve d4rl dataset from berkley servers
    dataset = d4rl.qlearning_dataset(envn)

    envn.reset()

    print(f"***************{env}**************")
