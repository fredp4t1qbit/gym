from gym import spaces
import numpy as np
import os, json, gym, sys, random, inspect

class KBlockerEnv(gym.Env):
    def __init__(self, config_path=''):
        if(not config_path):
            config_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            config_path = config_folder + '/config'

        with open(config_path) as data_file:
            self.config = json.load(data_file)

        self.__ep_ctr = 0

        # Every _SAMPLE_TRAJ_INTERVAL episodes, print one.
        self._PRINT_TRAJ_SAMPLES = False
        if self._PRINT_TRAJ_SAMPLES:
            self._traj_file = open('trajs.txt', 'w')
        self._SAMPLE_TRAJ_INTERVAL = 100
        self.__traj_sample_ctr = -1

        self.__num_bs = self.config['task']['num_blockers']
        self.__grid_y = self.config['task']['grid_y']
        self.__grid_x = self.__num_bs * 3 + 1
        self.__i2t_actions = [(1,0), (-1,0), (0,-1), (0,1)]  # right, left, back, forward

        self._reset()
        self._seed()

        self.__attacks = []
        self._NUM_ACTIONS = 4
        self._WIN_REWARD = self.config['task']['win_reward']
        self._PLAY_REWARD = self.config['task']['play_reward']
        self._MAX_EPISODE = 20

        # 0: right, 1: left, 2: down, 3: up
        self.action_space = spaces.Discrete(self._NUM_ACTIONS ** (self.__num_bs + 1))

        lows = np.array([0, 0] * (self.__num_bs + 1))
        highs = np.array([self.__grid_x, self.__grid_y] * (self.__num_bs + 1))
        self.observation_space = spaces.Box(lows, highs)

    def __decode__action__(self, action):
        var = []
        for i in range(self.__num_bs + 1):
            var.append(action % self._NUM_ACTIONS)
            action = int(action / self._NUM_ACTIONS)
        return var

    def __encode__action__(self, action_array):
        var = 0
        factor = 1
        for a in action_array:
            var += a * factor
            factor *= self._NUM_ACTIONS

        return var


    def __encode__(self, coord):
        return coord[1] * self.__grid_x + coord[0]


    def __coord_after__(self, action, coord):
        # a blocker is blocking (3 is up)
        if action == 3 and coord[1] is self.__grid_y -2 and coord[0] is not self.__gap:
            return coord

        new_coord = [sum(x) for x in zip(coord, self.__i2t_actions[action])]
        new_coord[0] = min(max(0, new_coord[0]), self.__grid_x - 1)
        new_coord[1] = min(max(0, new_coord[1]), self.__grid_y - 1)

        tnew = tuple(new_coord)
        # don't move if someone's there already.
        return coord if (tnew in self.__coords) else tnew

    def __defend__(self):

        if self.__gap not in self.__attacks:
            return self.__gap

        for i in range(self.__gap - 3, -1, -3):
            if i not in self.__attacks:
                return i
        for i in range(self.__gap + 3, self.__grid_x, 3):
            if i not in self.__attacks:
                return i

        return self.__gap

    def _seed(self, seed=None):
        random.seed(seed)
        return seed

    def _reset(self):
        self.__ep_ctr = 0

        if self._PRINT_TRAJ_SAMPLES and not self.__traj_sample_ctr % self._SAMPLE_TRAJ_INTERVAL:
            self._traj_file.write('-----------------')
            self._traj_file.write(os.linesep)
            self._traj_file.flush()

        self.__traj_sample_ctr += 1

        self.__gap = 0

        # Initialize the agents randomly
        self.__coords = [(r, 0) for r in random.sample(range(self.__grid_x), self.__num_bs + 1)]

        return self._get_encoded_state()


    def _get_encoded_state(self):
        """
        i.e.
        >>> self.__coords
                Out: [(2, 0), (1, 0)]
        >>> self._get_encoded_state()
                Out: (2, 0, 1, 0)

        :return: a tuple of length 2x the number of blocks (which is the number of blockers + 1)
        """
        return sum(self.__coords, ())

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # If won, stay for one move here (regardless of action), return no reward and declare win
        won_already = any(c[1] == self.__grid_y - 1 for c in self.__coords)
        if won_already:
            state = self._get_encoded_state()
            return state, 0, True, {}

        self.__ep_ctr += 1

        # Let the blockers defend
        self.__attacks = [x[0] for x in self.__coords if x[1] == self.__grid_y - 2]
        self.__gap = self.__defend__()

        # Agents attack again
        sep_actions = self.__decode__action__(action)
        for i in range(len(self.__coords)):
            self.__coords[i] = self.__coord_after__(sep_actions[i], self.__coords[i])

            if self._PRINT_TRAJ_SAMPLES and not self.__traj_sample_ctr % self._SAMPLE_TRAJ_INTERVAL:
                self._traj_file.write(str(self.__coords[i]) + ' # ')

        if self._PRINT_TRAJ_SAMPLES and not self.__traj_sample_ctr % self._SAMPLE_TRAJ_INTERVAL:
            self._traj_file.write(os.linesep)

        won = any(c[1] == self.__grid_y - 1 for c in self.__coords)

        state = self._get_encoded_state()

        rew = self._WIN_REWARD if won else self._PLAY_REWARD
        return state, rew, False, {}
