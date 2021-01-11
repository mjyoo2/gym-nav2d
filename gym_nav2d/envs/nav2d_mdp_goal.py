from gym import error, spaces, utils
import numpy as np
import math

from gym_nav2d.envs.nav2d_env import Nav2dEnv


class Nav2dMDPGoal(Nav2dEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        Nav2dEnv.__init__(self)
        # observation only distance 0..1
        self.observation_space = spaces.Box(np.array([0, 0, 0]), np.array([1, 1, 1]), dtype=np.float32)

    def goal_setting(self, goal):
        self.goal_x = goal[0]
        self.goal_y = goal[1]

    def _observation(self):
        return np.array([self.agent_x, self.agent_y, self._distance()])

    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, 2):
            normalized_obs.append(obs[i]/255*2-1)
        normalized_obs.append(obs[-1]/360.62)
        return np.array(normalized_obs)

    def reset(self):
        # Changing start point and fixed goal point
        self.count_actions = 0
        self.positions = []
        self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale,
                  self.goal_y*self.scale)
        obs = self._observation()

        return self._normalize_observation(obs)

    def step(self, action):
        self.count_actions += 1
        self._calculate_position(action)

        # calulate new observation
        obs = self._observation()

        # done for rewarding
        done = bool(obs[2] <= self.eps)
        rew = 0
        if not done:
            rew += self._step_reward()
        else:
            rew += self._reward_goal_reached()

        # break if more than max_steps actions taken
        done = bool(obs[2] <= self.eps or self.count_actions >= self.max_steps)

        normalized_obs = self._normalize_observation(obs)

        info = "Debug:" + "actions performed:" + str(self.count_actions) + ", act:" + str(action[1]) + "," + str(
            action[0]) + ", dist:" + str(normalized_obs[2]) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

        info = {'des': info[0], 'goal': info[1]}

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        return normalized_obs, rew, done, info
