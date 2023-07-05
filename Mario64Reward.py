import numpy as np
from ultralytics import YOLO

class Mario64Reward():
    YOLO_MODEL = None

    DEFAULT_STEP_REWARD = -1
    BACKWARDS_PENALTY = 5

    SCREEN_CENTER_X = 640/2
    SCREEN_CENTER_Y = 480/2

    YELLOW_COIN_DISCOUNT = 0.0005
    RED_COIN_DISCOUNT = 0.0005
    STAR_DISCOUNT = 0.0005

    AREA_DISCOUNT = 0.0005

    YELLOW_COIN_REWARD = 10
    RED_COIN_REWARD = 10
    STAR_REWARD = 10
    
    LOOK_FOR_YELLOW_COINS = True
    LOOK_FOR_RED_COINS = True
    LOOK_FOR_STARS = False

    previous_reward = 0

    cur_damage = 0

    found_collectable_last_step = False

    DISTANCE_DISCOUNT = 0.005

    def __init__(self):
        self.model = YOLO('YoloV8/weights/best.pt')

    def reset(self):
        self.previous_reward = 0
        self.found_collectable_last_step = False
        self.cur_damage = 0

    def get_reward(self, frame: np.array, damage: int, coins: int) -> float:                    
        # Reward the agent for being close to collectables
        reward_to_return, found_collectible, num_collectables = self.find_collectables(frame)

        # Moving away from collectables
        if reward_to_return < self.previous_reward and found_collectible and self.found_collectable_last_step:
            reward_to_return -= self.BACKWARDS_PENALTY

        self.previous_reward = reward_to_return
        self.found_collectable_last_step = found_collectible

        damage_penalty = 2 * damage

        if damage > self.cur_damage:
            damage_penalty *= 2

        if damage < self.cur_damage:
            damage_penalty *= -2
            
        self.cur_damage = damage

        reward_to_return -= damage_penalty

        reward_to_return += coins

        return reward_to_return, num_collectables
    
    def find_collectables(self, frame: np.array) -> tuple:
        reward = 0
        found_collectible = False
        num_collectables = 0

        preds = self.model.predict(frame[:,:,::-1])

        for preds in preds[0].cpu().boxes.data.numpy():
            area = (preds[2] - preds[0]) * (preds[3] - preds[1]) * self.AREA_DISCOUNT

            confidence = preds[4]

            object_class = int(preds[5])

            if confidence < 0.5:
                continue
            
            found_collectible = True
            if object_class == 0 and self.LOOK_FOR_STARS:
                continue
            elif object_class == 1 and self.LOOK_FOR_RED_COINS:
                continue
            elif object_class == 2 and self.LOOK_FOR_YELLOW_COINS:
                num_collectables += 1
                reward += self.YELLOW_COIN_REWARD * area
                print("reward", reward)

        if num_collectables == 0:
            reward = self.DEFAULT_STEP_REWARD

        return reward, found_collectible, num_collectables

    def get_distance_to_center(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt((np.mean([x1, x2]) - self.SCREEN_CENTER_X)**2 + (np.mean([y1, y2]) - self.SCREEN_CENTER_Y)**2)