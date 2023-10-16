from gymnasium import spaces
import numpy as np
import gymnasium as gym
import cv2
import mss
import time
import pyautogui
from Mario64MultiInputReward import Mario64MultiInputReward
from Mario64CoinParser import Mario64CoinParser
from Constants import Constants

N_DISCRETE_ACTIONS = len(Constants.DISCRETE_ACTIONS)
NUM_ACTION_HISTORY = 16

class Mario64MultiInputEnv(gym.Env):    
    def __init__(self):
        super(Mario64MultiInputEnv, self).__init__()

        self.set_pyautogui_pause()

        self.action_space: spaces.Discrete = spaces.Discrete(N_DISCRETE_ACTIONS)

        spaces_dict: dict = {
            'image': spaces.Box(
                        low=0, 
                        high=255, 
                        shape=(Constants.SCREEN_HEIGHT, 
                               Constants.SCREEN_WIDTH, 
                               3), 
                        dtype=np.uint8
                    ),
            'collectables': spaces.Box(
                        low=0,
                        high=255,
                        shape=(1,),
                        dtype=np.uint8
                    ),
            'damage': spaces.Box(
                        low=0,
                        high=9,
                        shape=(1,),
                        dtype=np.uint8
                    ),
            'previous_actions': spaces.Box(
                        low=0, 
                        high=1, 
                        shape=(NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS, 1), 
                        dtype=np.uint8
                    )
        }

        self.observation_space: spaces.Dict = spaces.Dict(spaces_dict)

        self.iteration: int = 1
        self.cur_damage: int = 0
        self.num_of_coins: int = 0
        self.cur_max_reward: int = -100

        self.sped_up_game: int = False

        self.action_history: list = []

        self.reward: Mario64MultiInputReward = Mario64MultiInputReward()
        self.coin_parser: Mario64CoinParser = Mario64CoinParser()

        self.sct: mss.mss = mss.mss()

    def get_state(self):
        sct_img = self.sct.grab(Constants.MONITOR)
        return cv2.cvtColor(np.asarray(sct_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    def step(self, action: int):
        t0 = time.time()
        self.iteration += 1
        episode_over = False

        self.action_history.append(action)
        self.take_action(action)
        #! time.sleep(0.5) # Comment above line and uncomment this line to take control of the game

        state = self.get_state()
        damage = self.get_damage(state)

        self.num_of_coins = self.coin_parser.get_num_of_coins(state, self.num_of_coins)

        if damage == 7 or self.iteration % 500 == 0:
            episode_over = True

        reward, num_of_collectables = self.reward.get_reward(state, damage, self.num_of_coins)

        self.cur_max_reward = max(self.cur_max_reward, reward)

        t_end = time.time()

        current_fps = round(((1 / (t_end - t0)) * 10), 0) 
        
        if not episode_over:
            print(f'Iteration: {self.iteration} | FPS: {current_fps} | Coins: {self.num_of_coins} | Damage: {damage} | Reward: {reward} | Action: {Constants.ACTION_NUM_TO_WORD[action]}')
        else:
            print(f'FPS: {current_fps} | Reward: {reward} | Max Reward: {self.cur_max_reward}')

        observation = {
            'image': state,
            'collectables': num_of_collectables,
            'damage': damage,
            'previous_actions': self.oneHotPrevActions(self.action_history)
        }

        return observation, reward, episode_over, False, {}
    
    def reset(self, seed=None, options=None):
        self.take_action(0)
        super().reset()
        self.get_random_reset_point()
        if self.iteration == 1 and not self.sped_up_game:
            for _ in range(10):
                pyautogui.press('F11')
                time.sleep(0.1)
            self.sped_up_game = True

        self.action_history = []
        self.cur_max_reward = -100
        self.num_of_coins = 0
        self.reward.reset()

        observation = {
            'image': self.get_state(),
            'collectables': 0,
            'damage': 0,
            'previous_actions': self.oneHotPrevActions(self.action_history)
        }

        return observation, {}

    def oneHotPrevActions(self, action_history: list):
        oneHot = np.zeros(shape=(NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS))
        if not action_history:
            return oneHot.reshape((NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS, 1))
        
        for i in range(NUM_ACTION_HISTORY):
            if len(action_history) >= (i + 1):
                oneHot[i][action_history[-(i + 1)]] = 1

        return oneHot.reshape((NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS, 1))

    def take_action(self, action: int):
        if action == 0:
            self.normalize_pyautogui_pause()
            pyautogui.keyUp('up')
            pyautogui.keyUp('down')
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            self.set_pyautogui_pause()
        elif action == 1:
            pyautogui.keyUp('down')
            pyautogui.keyDown('up')
        elif action == 2:
            pyautogui.keyUp('up')
            pyautogui.keyDown('down')
        elif action == 3:
            pyautogui.keyUp('right')
            pyautogui.keyDown('left')
        elif action == 4:
            pyautogui.keyUp('left')
            pyautogui.keyDown('right')
        elif action == 5:
            pyautogui.keyDown('shift')
            pyautogui.keyUp('shift')
        elif action == 6:
            pyautogui.keyDown('ctrl')
            pyautogui.keyUp('ctrl')
        elif action == 7:
            pyautogui.keyDown('z')
            pyautogui.keyUp('z')
        # elif action == 8:
        #     keyboard.press_and_release('x')
        # elif action == 9:
        #     keyboard.press_and_release('c')

    def get_damage(self, state):
        health = state[42:88, 255:305]
        damage_likelihood = []

        if not np.any(state):
            return 7

        for i in range(1, 8):
            damage_likelihood.append(np.sum(health == Constants.DAMAGE_TAKEN[str(i)][...,::-1]))

        damage = np.argmax(damage_likelihood)
        if damage_likelihood[damage] != 6900:
            return 0
        return damage + 1

    def get_random_reset_point(self):
        random_reset_point = np.random.randint(0, 6)
        pyautogui.press(f'{random_reset_point}')
        time.sleep(0.1)
        pyautogui.press(f'F7')

    def render_state(self, state):                
        cv2.imshow('debug-render', state)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self):
        return 42
    
    def set_pyautogui_pause(self) -> None:
        pyautogui.PAUSE = 0.075

    def normalize_pyautogui_pause(self) -> None:
        pyautogui.PAUSE = 0.035