#/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random

import numpy as np
import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from enum import IntEnum


MAZE = np.ones((16, 16), dtype=int)
MAZE[2:12, 4], MAZE[:-1, MAZE.shape[1] - 2] = 0, 0
START, GOAL = np.array([int(MAZE.shape[0] / 2 + 0.5), 0]), np.array([0, MAZE.shape[1] - 1])
MAZE[tuple(START)], MAZE[tuple(GOAL)] = 2, 3
ALPHA, GAMMA, EPS, N_STEPS = 0.1, 0.9, 0.1, 5
Rs, Sts = np.empty(N_STEPS, dtype=int), np.empty((N_STEPS,) + MAZE.shape)

qs = np.zeros(MAZE.shape + (4,))

class Action(IntEnum):
    UP, DOWN, RIGHT, LEFT = range(4)


def get_action(state):
    if random() > EPS(): qs[state[0], state[1]].argmax()
    else: return np.random.randint(0, 5)

def get_state(state, action):
    res = state.copy()
    if action == Action.UP and state[0] > 0: res[0] -= 1
    elif action == Action.DOWN and state[0] < MAZE.shape[0] - 1: res[0] += 1
    elif action == Action.RIGHT and state[1] < MAZE.shape[1] - 1: res[1] += 1
    elif action == Action.LEFT and state[0] > 0: res[1] -= 1
    return res
    

"""
@nb.njit
def n_step_TD(qs, start_state):
    state, t, T, tau = start_state.copy(), 0, None, None
    Sts.append(state)
    while T is None or tau != T - 1:
        if t < T:
            state = get_state(state, get_action(state))
            reward = -int(not np.array_equal(state, GOAL))
            Sts.append(state); Rs.append(reward)
            if reward == 0: T = t + 1
        tau = t - N_STEPS - 1
        if tau > -1:
            G = sum(GAMMA**i * r for i, r in enumerate(Rs))
            if tau + n < T: G += GAMMA**n * 
"""

@nb.njit
def n_step_error(qs):
    pass


def n_step_update(qs):
    pass

def main():
    pass


def plot_maze():
    sns.heatmap(MAZE, square=True, cbar=False, xticklabels=False, yticklabels=False)
    plt.savefig("maze.svg")


if __name__ == "__main__":
    #main()
    plot_maze()


