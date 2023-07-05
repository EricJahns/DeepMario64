import numpy as np

class Mario64CoinParser():
    def __init__(self):
        self.coins = [self.one, self.two, self.three, self.four, self.five, self.six, self.seven, self.eight, self.nine]

    def get_num_of_coins(self, frame, cur_num_of_coins):
        frame = frame[11:43, 395:424][:,:,::-1][...,0]

        for coord in self.coins[cur_num_of_coins]:
            if frame[coord[0], coord[1]] != 255:
                break

        return cur_num_of_coins

    one = np.array([
        [1, 19],
        [30, 16],
        [9, 7],
        [14, 20],
        [22, 14],
        [16, 16]
    ])

    two = np.array([
        [8, 2],
        [2, 15],
        [26, 9],
        [24, 20],
        [7, 22],
        [19, 11]
    ])

    three = np.array([
        [7, 22],
        [9, 4],
        [3, 16],
        [15, 16],
        [22, 5]
    ])

    four = np.array([
        [28, 15],
        [2, 16],
        [22, 3],
        [21, 24],
        [18, 16]
    ])

    five = np.array([
        [8, 7],
        [3, 22],
        [22, 22],
        [28, 10]
    ])

    six = np.array([
        [22, 3],
        [22, 25],
        [12, 12],
        [29, 14],
        [2, 14]
    ])

    seven = np.array([
        [12, 4],
        [1, 10],
        [1, 17],
        [1, 24],
        [29, 11],
        [14, 17]
    ])

    eight = np.array([
        [16, 14],
        [7, 4],
        [2, 13],
        [24, 4],
        [24, 23]
    ])

    nine = np.array([
        [2, 13],
        [7, 23],
        [11, 4],
        [17, 15],
        [28, 15],
        [17, 5]
    ])
        
    

    