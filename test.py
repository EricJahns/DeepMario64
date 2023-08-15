from gym.wrappers import TimeLimit
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from Mario64Env import Mario64Env
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
import torch
from imitation.algorithms import bc

TRAIN_TIMESTEPS = 10_000

RESUME_TRAINING = True

MODEL_DIR = "./models/recurrentppo"

MODEL_NAME = "mario64"

DEVICE = 'cuda'

POLICY = "CnnLstmPolicy"

# Create a single environment for training with SB3
env = Mario64Env()
env = TimeLimit(env, max_episode_steps=500)

# Create a vectorized environment for training with `imitation`


# Option A: use a helper function to create multiple environments
def _make_env():
    """Helper function to create a single environment. Put any logic here, but make sure to return a RolloutInfoWrapper."""
    _env = Mario64Env()
    _env = TimeLimit(_env, max_episode_steps=500)
    _env = RolloutInfoWrapper(_env)
    return _env

def setup_RecurrentPPO(env: Mario64Env):
    config = {
        "policy": POLICY,
        "n_steps": 500,
        "learning_rate": 3e-6,
        "gamma": 0.99,
        "batch_size": 256,
        "n_epochs": 10,
        "normalize_advantage": True,
        "policy_kwargs": dict(activation_fn=torch.nn.LeakyReLU, net_arch=dict(vf=[64, 64, 32], pi=[64, 64, 32])),
        "verbose": 2,
        "seed": 42,
        "tensorboard_log": "./logs/ppo_mario64",
        "device": 'cuda'
    }

    return RecurrentPPO(config["policy"],
                        env,
                        n_steps=config["n_steps"],
                        learning_rate=config["learning_rate"],
                        gamma=config["gamma"],
                        batch_size=config["batch_size"],
                        n_epochs=config["n_epochs"],
                        normalize_advantage=config["normalize_advantage"],
                        policy_kwargs=config["policy_kwargs"],
                        verbose=config["verbose"],
                        seed=config["seed"],
                        tensorboard_log=config["tensorboard_log"],
                        device=config["device"])


venv = DummyVecEnv([_make_env for _ in range(4)])


# Option B: use a single environment
env = Mario64Env()
venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap a single environment -- only useful for simple testing like this

model = setup_RecurrentPPO(env)

expert = model.load(f"{MODEL_DIR}/{MODEL_NAME}_33", env=env)

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    expert.get_env(),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

bc_trainer.train(n_epochs=1)