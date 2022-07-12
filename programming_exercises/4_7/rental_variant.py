#/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.random import poisson
from numpy import zeros, unique, array
from numba import njit
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

DISTS = []
for i in range(3):
    size = 10000
    pois = poisson(2 + i, size)
    uniq, counts = unique(pois, return_counts=True)
    DISTS.append(counts / size)
LAMBDA_REQ_0, LAMBDA_REQ_1 = DISTS[1], DISTS[2]
LAMBDA_RET_0, LAMBDA_RET_1 = DISTS[1], DISTS[0]
MOVE_REWARD, RENT_REWARD, PARKING_REWARD = -2, 10, -4
GAMMA, THETA = 0.9, 0.1
MAX_CARS, MAX_MOVE, MAX_FREE_PARKING = 20, 5, 10

vs = zeros((MAX_CARS+1,MAX_CARS+1))
pi = vs.copy().astype(int)


def save_policy():
    save_policy.counter += 1
    ax = sns.heatmap(pi, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('policy_variant'+str(save_policy.counter)+'.svg')
    plt.close()
    
def save_value():
    save_value.counter += 1
    ax = sns.heatmap(vs, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('value_variant'+ str(save_value.counter)+'.svg')
    plt.close()

save_policy.counter = 0
save_value.counter = 0


@njit
def apply_action(state: array, action: int) -> float:
    return array([max(0, min(MAX_CARS, state[0] - action)), max(0, min(MAX_CARS, state[1] + action))])

@njit
def get_new_state_indexes(state: array, action0: int, action1: int) -> float:
    return [max(0, min(MAX_CARS, state[0] + action0)), max(0, min(MAX_CARS, state[1] + action1))]

@njit
def get_v(vs: array, state: array, action: int) -> float:

    parking_cost = (int(state[0] > MAX_FREE_PARKING) + int(state[0] > MAX_FREE_PARKING)) * PARKING_REWARD
    v = abs(action - int(action > 0)) * MOVE_REWARD + parking_cost
    state = apply_action(state, action)

    for i0, req_0 in enumerate(LAMBDA_REQ_0):
        for j0, ret_0 in enumerate(LAMBDA_RET_0):
            for i1, req_1 in enumerate(LAMBDA_REQ_1):
                for j1, ret_1 in enumerate(LAMBDA_RET_1):
                    prob = req_0 * req_1 * ret_0 * ret_1
                    i0, i1 = min(state[0], i0), min(state[1], i1)
                    i, j = get_new_state_indexes(state, j0 - i0, j1 - i1)
                    rew = (i0 + i1) * RENT_REWARD
                    v += prob * (rew + GAMMA * vs[i, j])
    return v

@njit
def get_pol(vs: array, state: array):
    l_action = min(MAX_MOVE, state[0])
    f_action = -min(MAX_MOVE, state[1]) 

    best, best_val = f_action, get_v(vs, [state[0], state[1]], f_action)
    for i in range(min(0, f_action + 1), l_action + 1):
        val = get_v(vs, [state[0], state[1]], i)
        if best_val < val:
            best, best_val = i, val
    return best


def main():
    while True:
        # policy evaluation
        while True:
            delta = 0
            with trange(vs.shape[0]) as t:
                for i in t:
                    for j in range(vs.shape[1]):
                        t.set_description(
                            "policy evaluation sweep no.%i, state: [%2i, %2i]" % (save_value.counter+1, i, j))
                        v, a = vs[i,j], pi[i,j]
                        vs[i, j] = get_v(vs, array([i,j]), a)
                        delta = max(delta, abs(v - vs[i,j]))
                print(f"delta: {delta}, THETA: {THETA}")
                if delta < THETA: break

        # policy improvement
        print("pol improv")
        policy_stable = True
        with trange(vs.shape[0]) as t:
            for i in t:
                for j in range(vs.shape[1]):
                    t.set_description(
                        "policy improvement sweep no.%i, state: [%2i, %2i]" % (save_policy.counter+1, i, j))
                    old_action = pi[i,j]
                    pi[i,j] = get_pol(vs, array([i,j]))
                    if old_action != pi[i,j]:
                        policy_stable = False
        save_value()
        save_policy()
        if policy_stable: break

if __name__ == "__main__":
    main()

