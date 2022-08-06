#/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym 
from gym import spaces
from stable_baselines3 import PPO

import numpy as np


import racetracks as rt

rng = np.random.default_rng()

class RacetrackEnv(gym.Env):
    def __init__(self, size=rt.RACETRACK.shape):
        self.observation_space = spaces.Dict({
            'position': spaces.Box(np.zeros(2), np.array(size)-1, dtype=int),
            'velocity': spaces.Box(0, 5, shape=(2,), dtype=int)
        })
        self.action_space = spaces.Discrete(9)
        self._action_to_direction = {
            0: np.array([-1, -1]),
            1: np.array([-1,  0]),
            2: np.array([-1,  1]),
            3: np.array([ 0, -1]),
            4: np.array([ 0,  0]),
            5: np.array([ 0,  1]),
            6: np.array([ 1, -1]),
            7: np.array([ 1,  0]),
            8: np.array([ 1,  1]),
        }
        self._agent_location = rng.choice(rt.ON_START)
        self._velocity = np.zeros(2, dtype=int)

    def get_direction(self, action):
        return self._action_to_direction[action]

    def is_on_grid(self, state_coord, grid):
        for f in grid:
            if np.array_equal(state_coord, f): return True
        return False

    def is_off_track(self, state_coord):
        if state_coord[0] < 0 or state_coord[1] > rt.RACETRACK.shape[1] - 1:
            return True
        return self.is_on_grid(state_coord, rt.OFF_TRACK)

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - rt.ON_FINISH).min()}

    def _get_obs(self):
        return {'position': self._agent_location, 'velocity': self._velocity}

    def reset(self, seed=None, return_info=False, options=None):
        self._agent_location = rng.choice(rt.ON_START)
        self._velocity = np.zeros(2, dtype=int)
        return self._get_obs()

    def step(self, action):
        direction = self._action_to_direction[action]
        self._velocity = np.clip(self._velocity + direction, 0, 5)
        self._agent_location[0] -= self._velocity[0]
        self._agent_location[1] += self._velocity[1]
        if self.is_off_track(self._agent_location):
            self._agent_location = rng.choice(rt.ON_START).copy()
        reward = -int(not self.is_on_grid(self._agent_location, rt.ON_FINISH))
        done = reward == 0
        info, observation = self._get_info(), self._get_obs()
        return observation, reward, done, info


if __name__ == "__main__":
    env = RacetrackEnv()
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=25_000)
    print("start predict")
    for _ in range(10):
        done, i, obs = False, 0, env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            #print(obs, rewards, action, env.get_direction(action))
            i += 1
        print(i)

