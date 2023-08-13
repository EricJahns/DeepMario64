import cv2
import mss

class Constants:
    #! Adjust me to crop the screen, use setup.ipynb to find the correct values
    SCREEN_X: int = 962
    SCREEN_Y: int = 536

    SCREEN_WIDTH: int = 636
    SCREEN_HEIGHT: int = 424

    YOLO_MODEL_DIR: str = 'YoloV8/weights/best.pt'

    DISCRETE_ACTIONS: dict = {
        'release_wasd': 'release_wasd',
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

    ACTION_NUM_TO_WORD: dict = {
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

    DAMAGE_TAKEN: dict = {
        '1': cv2.imread('./damage_assets/1_damage.png'),
        '2': cv2.imread('./damage_assets/2_damage.png'),
        '3': cv2.imread('./damage_assets/3_damage.png'),
        '4': cv2.imread('./damage_assets/4_damage.png'),
        '5': cv2.imread('./damage_assets/5_damage.png'),
        '6': cv2.imread('./damage_assets/6_damage.png'),
        '7': cv2.imread('./damage_assets/7_damage.png')
    }

    #! Adjust me to change the monitor, use setup.ipynb to find the correct values
    MONITOR_NUMBER: int = 1

    CUR_MONITOR = mss.mss().monitors[MONITOR_NUMBER]

    MONITOR: dict = {
        "top": CUR_MONITOR["top"] + SCREEN_Y, 
        "left": CUR_MONITOR["left"] + SCREEN_X, 
        "width": SCREEN_WIDTH, 
        "height": SCREEN_HEIGHT,
        "mon": CUR_MONITOR
    }