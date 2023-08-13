from Mario64Env import Mario64Env
# from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import os
import glob
import torch
from termcolor import cprint

TRAIN_TIMESTEPS = 10_000

RESUME_TRAINING = True

MODEL_DIR = "./models/recurrentppo"

MODEL_NAME = "mario64"

DEVICE = 'cuda'

POLICY = "CnnLstmPolicy"

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
            
def train(model, env: Mario64Env):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    iteration = 0
    if RESUME_TRAINING:
        iteration = get_last_iteration_num(MODEL_DIR)
        if iteration != 0:
            cprint(f"Resuming training from last saved model iteration {iteration}...", "yellow")
            model.load(f"{MODEL_DIR}/{MODEL_NAME}_{iteration}", env=env)
        iteration += 1
        
    while True:
        model.learn(total_timesteps=TRAIN_TIMESTEPS,
                    log_interval=4,
                    progress_bar=True,
                    tb_log_name="ppo_mario64")
    
        model.save(f"{MODEL_DIR}/{MODEL_NAME}_{iteration}")
        iteration += 1

    
def get_last_iteration_num(dir_path: str) -> int:
    """
    Get the last iteration number from a directory of saved models.

    Used for continuing training from a saved model.
    """
    models = glob.glob(os.path.join(dir_path, "*.zip"))
    if len(models) == 0:
        return 0
    
    return max([int(model.split("_")[-1].split(".")[0]) for model in models])
        
if __name__ == "__main__":
    env = Mario64Env()
    model = setup_RecurrentPPO(env)
    # print(model.polic y)
    train(model, env)