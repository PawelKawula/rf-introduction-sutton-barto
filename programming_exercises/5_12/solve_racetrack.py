#/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
from copy import copy, deepcopy
from random import random

import numpy as np
import numpy.random as nprand
import numba
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import racetracks

MAKE_PLOTS, MAKE_ARRAYS, MAKE_PREFS, PRINT_RESULTS = True, True, True, True

GAMMA, EPS, PROB_HALT = 0.9, 0.3, 0.1
MAX_VELOCITY, N_EPISODES, BASE_INCREMENT = 5, int(1e6), int(1)
#print(BASE_INCREMENT)

STATE_SHAPE = (racetracks.RACETRACK.shape + (MAX_VELOCITY + 1, MAX_VELOCITY + 1))
qs = np.zeros(STATE_SHAPE + (3, 3))
ns = np.zeros_like(qs, dtype=int)
pi = np.ones(STATE_SHAPE + (2,), dtype=int)
rng = nprand.default_rng()
lock = threading.Lock()
START = rng.choice(racetracks.ON_START)

@numba.njit
def is_stat_later(state, action, timestamps_before, desc=False):
    for st, at, _ in timestamps_before:
        if np.array_equal(st, state) and np.array_equal(at, action):
            return True
    return False

@numba.njit
def get_action(pi, state_coord, velocity, learn):
    if learn and random() < EPS:
        return nprand.randint(-1, 2, size=2)
    else:
        return pi[state_coord[0], state_coord[1], velocity[0], velocity[1]]


@numba.njit
def is_off_track(state_coord):
    if state_coord[0] < 0 or state_coord[1] > racetracks.RACETRACK.shape[1] - 1:
        return True
    return is_on_grid(state_coord, racetracks.OFF_TRACK)


@numba.njit
def is_on_grid(state_coord, grid):
    for f in grid:
        if np.array_equal(state_coord, f): return True
    return False


@numba.njit
def generate_episode(pi, start_coord, learn=True, desc=False):
    timesteps, state_coord, velocity = numba.typed.List(), start_coord.copy(), np.array([0, 0])
    while not is_on_grid(state_coord, racetracks.ON_FINISH):
        action = get_action(pi, state_coord, velocity, learn)
        rand_act = not np.array_equal(action, pi[state_coord[0], state_coord[1],
                                                 velocity[0], velocity[1]])
        state = np.concatenate((state_coord, velocity))
        velocity = np.clip(velocity + action, 0, MAX_VELOCITY) #if random() > PROB_HALT else np.array([0, 0])
        state_coord[0] -= velocity[0]
        state_coord[1] += velocity[1]
        if is_off_track(state_coord):
            state_coord = racetracks.ON_START[nprand.randint(len(racetracks.ON_START))].copy()
            velocity = np.array([0, 0])
        reward = -int(not is_on_grid(state_coord, racetracks.ON_FINISH))
        timesteps.append((state, action, reward))
        if desc:
            print(state[:2],"->", state_coord, "vel:", velocity, "act_taken:",
                  action, "random_act:", rand_act, "reward:", reward)
    if desc: print("Episode len:", len(timesteps))
    return timesteps


@numba.njit
def policy_iteration(qs, ns, pi, episode, desc=False):
    g, ep_len = 0, len(episode)
    for t in range(ep_len):
        t = ep_len - t - 1
        state, action, reward = episode[t]
        g = GAMMA * g + reward
        if t == 0 or not is_stat_later(state, action, episode[:t]):
            #index, state = tuple(np.concatenate((state, action + 1))), tuple(state)
            i = np.concatenate((state, action + 1))
            #q, n = qs[index], ns[index] + 1
            q = qs[i[0], i[1], i[2], i[3], i[4], i[5]]
            n = ns[i[0], i[1], i[2], i[3], i[4], i[5]] + 1
            #qs[index], ns[index] = q + 1 / n * (g - q), n
            qs[i[0], i[1], i[2], i[3], i[4], i[5]] = q + 1 / n * (g - q)
            ns[i[0], i[1], i[2], i[3], i[4], i[5]] = n
            state_qs = qs[i[0], i[1], i[2], i[3]]
            state_qs_argmax = state_qs.argmax()
            #y, x = np.unravel_index(qs[state].argmax(), qs[state].shape)
            y, x = state_qs_argmax // qs.shape[-2], state_qs_argmax % qs.shape[-1]
            #pi[state] = np.array(y - 1, x - 1)
            #pi[i[0], i[1], i[2], i[3]] = np.array(y - 1, x - 1)
            if desc:
                print("t", t, "t", g, "state", state, "qs[state]:\n", state_qs, "\nand argmax:",
                      state_qs_argmax, "indexes: ", y, x, "pi before",
                      pi[i[0], i[1], i[2], i[3]], "pi now:", y-1, x-1,
                      "\nrandom_action:", not np.array_equal(pi[i[0], i[1], i[2], i[3]], action), "action",
                      action, "n", n, "reward", reward, "\n\n")
            pi[i[0], i[1], i[2], i[3], 0] = y - 1
            pi[i[0], i[1], i[2], i[3], 1] = x - 1
        elif desc: print(t, "is in the past")


def learn(start_coord, desc=False):
    policy_iteration(qs, ns, pi, generate_episode(pi, start_coord, True, desc), desc)


def make_plots():
    racemap = racetracks.RACETRACK.copy()
    for start_coord in racetracks.ON_START:
        for timestamp in generate_episode(pi, start_coord):
            state, action, reward = timestamp
            racemap[state[0], state[1]] += BASE_INCREMENT
    sns.heatmap(racemap, xticklabels=False,
            yticklabels=False, square=True)
    plt.savefig('rides.svg')
    preferences = np.sum(pi, axis=(2,3))
    up_pref, right_pref = preferences[:,:,0], preferences[:,:,1]
    sns.heatmap(up_pref, cbar=False, xticklabels=False,
            yticklabels=False, square=True)
    plt.savefig('up_pref.svg')
    sns.heatmap(right_pref, cbar=False, xticklabels=False,
            yticklabels=False, square=True)
    plt.savefig('right_pref.svg')

def make_arrays():
    with open("arrays.npy", "wb") as f:
        np.save(f, pi)
        np.save(f, qs)


def make_preferences():
    preferences = np.mean(pi, axis=(2,3))
    with open("preferences.npy", "wb") as f:
        np.save(f, preferences[:,:,0])
        np.save(f, preferences[:,:,1])


def main():
    start_coords = np.concatenate((racetracks.ON_TRACK, racetracks.ON_START))
    test = np.empty((len(racetracks.ON_TRACK), 2), dtype=int)
    if PRINT_RESULTS:
        for i, e in enumerate(tqdm(racetracks.ON_TRACK)):
            test[i,0] = len(generate_episode(pi, e))
    for _ in trange(N_EPISODES):
       learn(rng.choice(racetracks.ON_START), False)
    if PRINT_RESULTS:
        for i, e in enumerate(racetracks.ON_TRACK):
            test[i,1] = len(generate_episode(pi, e))
            print(test[i])
        print(f"avg: {np.mean(test[:,0])}, {np.mean(test[:, 1])}")
    if MAKE_ARRAYS: make_arrays()
    if MAKE_PREFS: make_preferences()
    if MAKE_PLOTS: make_plots()

def sandbox():
    start_coords = np.concatenate((racetracks.ON_TRACK, racetracks.ON_START))
    NUM_TRIES, tries_before, tries_after = 1000, [], []
    for start_coord in tqdm(start_coords):
        tries_before.append(len(generate_episode(pi, start_coord)))
    with open("arrays.npy", "rb") as f:
        pi = np.load(f)
        qs = np.load(f)
    for start_coord in tqdm(start_coords):
        tries_after.append(len(generate_episode(pi, start_coord)))
    for before, after in zip(tries_before, tries_after):
        print(f"BEFORE: {before}, AFTER: {after}")
    print(f"avg: {np.mean(tries_before)}, {np.mean(tries_after)}")
    make_plots()

def test():
    test = np.empty((len(racetracks.ON_START), 2), dtype=int)
    for i, e in enumerate(tqdm(racetracks.ON_START)):
        test[i, 0] = len(generate_episode(pi, e))
    with open("arrays.npy", "rb") as f:
        pi = np.load(f)
    for i, e in enumerate(tqdm(racetracks.ON_START)):
        test[i, 1] = len(generate_episode(pi, e))
    print(test, np.mean(test[:, 0]), np.mean(test[:, 1]))

#test()
main()
