import minari
import gym
import d4rl.gym_mujoco
import numpy as np
from minari.dataset import MinariDataset

envs=[
    "halfcheetah-random-v0",
    "hopper-random-v0",
	"walker2d-random-v0",
	"halfcheetah-medium-v0",
	"hopper-medium-v0",
	"walker2d-medium-v0",
	"halfcheetah-expert-v0",
	"hopper-expert-v0",
	"walker2d-expert-v0",
	"halfcheetah-medium-expert-v0",
	"hopper-medium-expert-v0",
	"walker2d-medium-expert-v0",
	"halfcheetah-medium-replay-v0",
	"hopper-medium-replay-v0",
	"walker2d-medium-replay-v0",
    ]

for env in envs:
    # retrieve d4rl dataset from berkley servers
    dataset = d4rl.qlearning_dataset(gym.make(env))

    contradictions = []
    for i in range(len(dataset['observations'])-1):
        if not np.equal(dataset['observations'][i+1], dataset['next_observations'][i]).all():
            contradictions.append(i)

    print(f"There are {len(contradictions)} instances where the observations do not match the next_observations.")

    print(f"There are {dataset['terminals'].sum()} terminal states.")
