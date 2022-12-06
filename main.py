import minari
import gym
import d4rl.gym_mujoco
import numpy as np
from minari.dataset import MinariDataset

envs=[
    # "halfcheetah-random-v0",
    "hopper-random-v0",
#	"walker2d-random-v0"
#	"halfcheetah-medium-v0"
#	"hopper-medium-v0"
#	"walker2d-medium-v0"
#	"halfcheetah-expert-v0"
#	"hopper-expert-v0"
#	"walker2d-expert-v0"
#	"halfcheetah-medium-expert-v0"
#	"hopper-medium-expert-v0"
#	"walker2d-medium-expert-v0"
#	"halfcheetah-medium-replay-v0"
#	"hopper-medium-replay-v0"
#	"walker2d-medium-replay-v0"
    ]

for env in envs:
    dataset_name = f"D4RL-{env[:-3]}_{env[-2:]}_Legacy-D4RL-dataset"

    # retrieve d4rl dataset from berkley servers
    env = d4rl.qlearning_dataset(gym.make(env))

    # convert d4rl dataset to minari dataset
    dataset = MinariDataset(
        dataset_name=dataset_name,
        algorithm_name="SAC (check code permalink for details)",
        seed_used=np.nan,
        code_permalink="https://github.com/Farama-Foundation/D4RL/wiki/Dataset-Reproducibility-Guide",
        author="Justin Fu",
        author_email="justinjfu@eecs.berkeley.edu",
        observations=np.vstack((env["observations"], env["next_observations"][-1])),  # env["observations"],
        actions=np.vstack((env["actions"], [0]*len(env["actions"][0]))),  # env["actions"],
        rewards=np.append(env["rewards"], 0),  # env["rewards"],
        terminations=np.append(env["terminals"], 1),  # env["terminals"],  assume last transition is terminal
        truncations=np.append(env["terminals"], 1)  # env["terminals"],
    )

    dataset_two = MinariDataset(
        dataset_name=dataset_name,
        algorithm_name="SAC (check code permalink for details)",
        seed_used=np.nan,
        code_permalink="https://github.com/Farama-Foundation/D4RL/wiki/Dataset-Reproducibility-Guide",
        author="Justin Fu",
        author_email="justinjfu@eecs.berkeley.edu",
        observations=env["observations"],
        actions=env["actions"],
        rewards=env["rewards"],
        terminations=env["terminals"],
        truncations=env["terminals"],
    )

    assert (dataset.observations[:-1] == env['observations']).all()
    assert (dataset.actions[:-1] == env['actions']).all()
    assert (dataset.rewards[:-1] == env['rewards']).all()
    assert (dataset.observations[1:] == env['next_observations']).all()
    assert (dataset.terminations[:-1] == env['terminals']).all()
    assert (dataset.truncations[:-1] == env['terminals']).all()

    #upload dataset to minari servers
    dataset.save()
    minari.upload_dataset(dataset_name)
