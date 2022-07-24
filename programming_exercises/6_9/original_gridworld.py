#/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random
from enum import IntEnum, unique

import numpy as np
import numpy.random as nprand
import numba as nb
import seaborn as sns
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

GAMMA, EPS, ALPHA, N_EPISODES = 0.9, 0.1, 0.1, 1000

PLOT_RESULTS, PLOT_PATH = True, True

START, GOAL = np.array([3,0]), np.array([3,7])
GRIDWORLD = np.zeros((7, 10), dtype=int)
YBOUND, XBOUND = GRIDWORLD.shape[0] - 1, GRIDWORLD.shape[1] - 1
# wind
GRIDWORLD[:, 3:9], GRIDWORLD[:, 6:8] = 1, 2

@unique
class Action(IntEnum):
    UP, DOWN, RIGHT, LEFT = range(4)


ACTION_LEN = len(Action)
qs = np.zeros(GRIDWORLD.shape + (len(Action),))

@nb.njit
def get_wind(state):
    return GRIDWORLD[state[0], state[1]]


@nb.njit
def get_state(state, action):
    res = state.copy() 
    if      action == Action.UP     and res[0] > 0:         res[0] -= 1
    elif    action == Action.DOWN   and res[0] < YBOUND:    res[0] += 1
    elif    action == Action.RIGHT  and res[1] < XBOUND:    res[1] += 1
    elif    res[1] > 0: res[1] -= 1
    res[0] = max(res[0] - get_wind(state), 0)
    return res


@nb.njit
def get_action(qs, state, rand=True):
    if random() > EPS or not rand: 
        return qs[state[0], state[1]].argmax()
    else: return nprand.randint(ACTION_LEN)


@nb.njit
def sarsa(qs, desc=False):
    state, action, timesteps = START.copy(), get_action(qs, START), nb.typed.List()
    while not np.array_equal(state, GOAL):
        state_plim = get_state(state, action) 
        action_plim = get_action(qs, state_plim)
        reward = -int(not np.array_equal(state_plim, GOAL))
        q, q_plim = qs[state[0], state[1], action], qs[state_plim[0], state_plim[1], action_plim]
        prev_max = qs[state[0], state[1]].argmax()
        qs[state[0], state[1], action] += ALPHA * (reward + GAMMA * q_plim - q)
        if desc:
            print(state, "->", state_plim, "\nqs[state]:", qs[state[0], state[1]],
                  "\nnow argmax:", qs[state[0], state[1]].argmax(), "action:", action,
                  "random action:", prev_max != action,
                  "reward:", reward, "action_plim", action_plim, "\n\n")
        timesteps.append((state, action, reward))
        state, action = state_plim, action_plim
    return timesteps


@nb.njit
def generate_episode(desc=False):
    state, timesteps = START.copy(), nb.typed.List()
    while not np.array_equal(state, GOAL):
        action = get_action(qs, state, True)
        state_plim = get_state(state, action)
        reward = -int(not np.array_equal(state_plim, GOAL))
        if desc: print(state, "->", state_plim, "action:", action)
        timesteps.append((state, action, reward))
        state = state_plim
    return timesteps


def plot_results(arr):
    fig, ax = plt.subplots()
    ax.plot(arr)
    plt.axhline(min(arr), color='red')
    plt.savefig("results.png")


def plot_gridworld():
    board = GRIDWORLD.copy()
    board[tuple(START)] = 5
    board[tuple(GOAL)] = 6
    sns.heatmap(board, square=True, cbar=False, cmap="YlGnBu", linewidths=.5)
    plt.savefig("gridworld.svg")


def plot_path():
    board = GRIDWORLD.copy() 
    for state, _, _ in generate_episode():
        board[tuple(state)] = 5
    board[tuple(START)] = 4
    board[tuple(GOAL)] = 6
    sns.heatmap(board, square=True, cbar=False, cmap="YlGnBu", linewidths=.5)
    plt.savefig("path.svg")


def main():
    arr = np.empty(N_EPISODES, dtype=int)
    for i in trange(N_EPISODES):
        arr[i] = len(sarsa(qs, False))
    if PLOT_RESULTS:    plot_results(arr)
    if PLOT_PATH:       plot_path()
    #print(generate_episode())


if __name__ == "__main__":
    main()
    #sarsa(qs, True)

