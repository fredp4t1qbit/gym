"""
Simulated Annealing

Copyright (C) 2018-2019 by Smriti Shyamal <smriti.shyamal@1qbit.com>

This code is a modification over the 2012-2013 code of Sergei Isakov (Google).
The rights and license is hence inherited from the original as GNU General
Public License (v3).
"""

import math
#import gym
#from gym import spaces, logger
#from gym.utils import seeding
import numpy as np

class Algorithm:
    def __init__(self):
        self.x = 4
        self.y = 3

    def step(self, x, y):
        sum = x + y
        return sum
