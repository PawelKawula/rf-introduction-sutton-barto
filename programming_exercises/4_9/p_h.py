#/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import zeros, double
from tqdm import trange, tqdm
from time import sleep
from random import random
from numba import njit
import matplotlib.pyplot as plt
from os import makedirs

P_H = 0.55
FINAL_STATE = 100
S_RANGE = range(FINAL_STATE+1)
vs, pi = zeros(FINAL_STATE+1, dtype=double), zeros(FINAL_STATE+1, dtype=int)
#vs[FINAL_STATE] = 1
GAMMA, THETA = 0.9, 1e-1

"""
def print_values():
    print("values:")
    for i, e in enumerate(vs):
        print("%3i: %4i" % (i, e), end="\t")
        if i % 5 == 0 and i != 0: print()

def print_policy():
    print("policy:")
    for i, e in enumerate(pi):
        print("%3i: %4i" % (i, e), end="\t")
        if i % 5 == 0 and i != 0: print()
"""

def save_value():
    dirpath = str(P_H).replace(".", "_")
    makedirs(dirpath, exist_ok=True)
    save_value.counter += 1
    fig, ax = plt.subplots()
    ax.plot(S_RANGE, vs)
    plt.xlabel("state")
    plt.ylabel("value")
    fig.savefig(dirpath + f"/value_{save_value.counter}.svg")

def save_policy():
    dirpath = str(P_H).replace(".", "_")
    makedirs(dirpath, exist_ok=True)
    save_policy.counter += 1
    fig, ax = plt.subplots()
    ax.scatter(S_RANGE, pi)
    plt.xlabel("state")
    plt.ylabel("stake")
    fig.savefig(dirpath + f"/policy_{save_policy.counter}.svg")

save_policy.counter = 0
save_value.counter = 0

@njit
def get_v(vs, state, action):
    state_win, state_lose = min(FINAL_STATE, state + action), max(0, state - action)
    v_win = P_H * (int(state + action >= FINAL_STATE) + GAMMA * vs[state_win])
    v_lose = (1 - P_H) * GAMMA * vs[state_lose]
    return v_win + v_lose
# why is this not working? 
# need to uncomment vs[FINAL_STATE] = 1
"""
@njit
def get_v(vs, state, action):
    state_win, state_lose = min(FINAL_STATE, state + action), max(0, state - action)
    return P_H * GAMMA * vs[state_win] + (1 - P_H) * GAMMA * vs[state_lose]
"""


@njit
def get_pol(vs, state):
    best, best_val = 1, get_v(vs, state, 1)
    for a in range(2, state + 1):
        val = get_v(vs, state, a)
        #if state < 5: print(state, a)
        if val > best_val:
            best, best_val = a, val
    return best
        
def main():
    while True:
        while True:
            delta = 0
            pbar = tqdm(S_RANGE, desc=f"Policy Evaluation no.{save_value.counter+1}")
            for state in pbar:
                pbar.set_description(
                    "policy evaluation sweep no.%i, state: %2i" % (save_value.counter+1, state))
                v = vs[state]
                vs[state] = get_v(vs, state, pi[state])
                delta = max(delta, abs(vs[state] - v))
            print(f"delta: {delta}, theta: {THETA}")
            if delta < THETA: break

        policy_stable = True
        pbar = tqdm(S_RANGE, desc=f"Policy Improvement no.{save_policy.counter+1}")
        for state in tqdm(S_RANGE):
            pbar.set_description(
                "policy improvement sweep no.%i, state: %2i" % (save_policy.counter+1, state))
            old_action = pi[state]
            pi[state] = get_pol(vs, state)
            if old_action != pi[state]: policy_stable = False
        save_value()
        save_policy()
        if policy_stable: break

main()
