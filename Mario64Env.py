from gymnasium import spaces
import numpy as np
import gymnasium as gym
import cv2
import mss
import time
import pyautogui
from Mario64Reward import Mario64Reward

DISCRETE_ACTIONS = {'release_wasd': 'release_wasd',
                    'w': 'run_forwards',                
                    's': 'run_backwards',
                    'a': 'run_left',
                    'd': 'run_right',
                    'shift': 'A',
                    'ctrl': 'B',
                    'z': 'Z',
                    # 'x': 'LT',
                    # 'c': 'RT'
                    }

ACTION_NUM_TO_WORD = {
    0: 'Release',
    1: 'Forwards',
    2: 'Backwards',
    3: 'Left',
    4: 'Right',
    5: 'Jump',
    6: 'Punch',
    7: 'Crouch',
    # 8: 'Left Trigger',
    # 9: 'Right Trigger'
}

DAMAGE_TAKEN = {
    '1': cv2.imread('./damage_assets/1_damage.png'),
    '2': cv2.imread('./damage_assets/2_damage.png'),
    '3': cv2.imread('./damage_assets/3_damage.png'),
    '4': cv2.imread('./damage_assets/4_damage.png'),
    '5': cv2.imread('./damage_assets/5_damage.png'),
    '6': cv2.imread('./damage_assets/6_damage.png'),
    '7': cv2.imread('./damage_assets/7_damage.png')
}

pyautogui.PAUSE = 0.10

###############################################
class Mario64Env(gym.Env):    
    def __init__(self):
        super(Mario64Env, self).__init__()

        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        screen_bounds = [960-536, 636]
        self.observation_space = spaces.Box(low=0, 
                                            high=255, 
                                            shape=(screen_bounds[0], screen_bounds[1], 3), 
                                            dtype=np.uint8)
        self.iteration = 1
        self.cur_damage = 0

        self.reward: Mario64Reward = Mario64Reward()

        self.cur_max_reward = -100
        self.sped_up_game = False
        self.sct = mss.mss()
    
    def grab_screen_shot(self):
        monitor = self.sct.monitors[1]
        sct_img = self.sct.grab(monitor)
        frame = cv2.cvtColor(np.asarray(sct_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        # self.render_frame(frame)
        return frame[536:960, 962:962+636]
    
    def render_frame(self, frame):                
        cv2.imshow('debug-render', frame)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

    def step(self, action: np.array):
        t0 = time.time()
        self.iteration += 1
        episode_over = False
    
        self.take_action(action)
        # time.sleep(0.1)

        frame = self.grab_screen_shot()
        damage = self.get_damage(frame)
        if damage == 7 or self.iteration % 500 == 0:
            episode_over = True

        self.cur_damage = damage

        reward, _ = self.reward.get_reward(frame, damage)

        self.cur_max_reward = max(self.cur_max_reward, reward)

        t_end = time.time()

        current_fps = round(((1 / (t_end - t0)) * 10), 0) 
        
        if not episode_over:
            print(f'Model: Iteration: {self.iteration} | FPS: {current_fps} | Damage: {self.cur_damage} | Reward: {reward} | Action: {ACTION_NUM_TO_WORD[action]}')
        else:
            print(f'Model: Reward: {reward} | FPS: {current_fps} | Max Reward: {self.cur_max_reward}')

        return frame, reward, episode_over, False, {}
    
    def get_damage(self, frame):
        health = frame[42:88, 255:305]
        damage_likelihood = []

        if not np.any(frame):
            return 7

        for i in range(1, 8):
            damage_likelihood.append(np.sum(health == DAMAGE_TAKEN[str(i)][...,::-1]))

        damage = np.argmax(damage_likelihood)
        if damage_likelihood[damage] != 6900:
            return 0
        return damage + 1
    
    def reset(self, seed=None, options=None):
        super().reset()
        self.take_action(0)
        self.get_random_reset_point()
        if self.iteration == 1 and not self.sped_up_game:
            for _ in range(10):
                pyautogui.press('F11')
                time.sleep(0.1)
            self.sped_up_game = True

        self.cur_max_reward = -100
        self.cur_damage = 0

        return self.grab_screen_shot(), {}
    
    def get_random_reset_point(self):
        random_reset_point = np.random.randint(0, 5)
        pyautogui.press(f'{random_reset_point}')
        time.sleep(0.1)
        pyautogui.press(f'F7')

    def take_action(self, action):
        if action == 0:
            pyautogui.keyUp('up')
            pyautogui.keyUp('down')
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
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
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self):
        return 42