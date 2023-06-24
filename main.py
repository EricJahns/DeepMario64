from Mario64Env import Mario64Env
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

TRAIN_TIMESTEPS = 10_000

USE_WANDB = True

DEVICE = 'cuda'

POLICY = "CnnLstmPolicy"

def train_PPO():
    env = Mario64Env()
    config = {
        "policy": POLICY,
        "n_steps": 250,
        "learning_rate": 3e-6,
        "gamma": 0.99,
        "batch_size": 256,
        "n_epochs": 10,
        "normalize_advantage": True,
        "verbose": 2,
        "tensorboard_log": "./logs/ppo_mario64",
        "device": 'cuda'
    }

    model = RecurrentPPO(config["policy"],
                env,
                n_steps=config["n_steps"],
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                normalize_advantage=config["normalize_advantage"],
                verbose=config["verbose"],
                tensorboard_log=config["tensorboard_log"],
                device=config["device"])
    
    model.load("./models/ppo_mario64_0", env=env)

    iteration = 1
    while True:
        model.learn(total_timesteps=TRAIN_TIMESTEPS,
                    log_interval=4,
                    progress_bar=True,
                    tb_log_name="ppo_mario64")
    
        model.save(f"./models/ppo_mario64_{iteration}")
        iteration += 1
        
if __name__ == "__main__":
    train_PPO()