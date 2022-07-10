#/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.random import poisson, randint
from numpy import zeros, unique, array
from numba import njit
from time import sleep
from tqdm import tqdm

def print_policy():
    for i in range(max_cars):
        for j in range(max_cars):
            print(abs(policy[i,j]), end=" ")
        print("")

@njit
def get_v(n_cars, a) -> float:
    cost = -abs(a * move_cost)
    v = 0 
    for i0, req_0 in enumerate(lambda_req_0):
        for j0, ret_0 in enumerate(lambda_ret_0):
            for i1, req_1 in enumerate(lambda_req_1):
                for j1, ret_1 in enumerate(lambda_ret_1):
                    i = max(0, min(max_cars, n_cars[0]-i0+j0))
                    j = max(0, min(max_cars, n_cars[1]-i1+j1))
                    prob = req_0 * req_1 * ret_0 * ret_1
                    rew = (i0 + i1) * rent_profit
                    v += prob * (cost + rew + gamma * Vs[i,j])
    return v

@njit
def get_pol(n_cars):
    bbx = max(0, min(max_cars, n_cars[0]-max_move))
    bby = max(0, min(max_cars, n_cars[1]+max_move))
    for i in range(-max_move+1, max_move+1):
        bix = max(0, min(max_cars, n_cars[0]-i))
        biy = max(0, min(max_cars, n_cars[1]+i))
        if Vs[bix,biy] > Vs[bbx, bby]:
            bbx, bby = bix, biy
            #bbx = max(0, min(max_cars, n_cars[0]-best))
            #bby = max(0, min(max_cars, n_cars[1]+best))
    #return Vs[n_cars[0]-best, n_cars[1]+best]
    return Vs[bbx, bby]

dists = []
for i in range(3):
    size = 10000
    pois = poisson(2+i, size)
    uniq, counts = unique(pois, return_counts=True)
    dists.append(counts / size)
lambda_req_0, lambda_req_1 = dists[1], dists[2]
lambda_ret_0, lambda_ret_1 = dists[1], dists[0]
bad_move_cost, move_cost = 100, 2
gamma, theta = 0.9, 0.1
rent_profit = 10
max_cars, max_move = 20, 5
Vs = zeros((max_cars+1,max_cars+1))
policy = randint(-max_move, max_move+1, size=(max_cars, max_cars))
policy_stable = False
print_policy()
while not policy_stable:
    # policy evaluation
    while True:
        delta = 0
        for i in range(max_cars):
            for j in range(max_cars):
                v, a = Vs[i,j], policy[i,j]
                ni, ny = max(0, min(max_cars, i-a)), max(0, min(max_cars, j+a))
                Vs[i, j] = get_v([ni,ny], a)
                delta = max(delta, abs(v - Vs[i,j]))
        print(f"eval sweep: {delta}")
        if delta < theta: break

    # policy improvement
    print("pol improv")
    policy_stable = True
    for i in range(max_cars):
        for j in range(max_cars):
            old_action = policy[i,j]
            policy[i,j] = get_pol(array([i,j]))
            if old_action != policy[i,j]: policy_stable = False
print_policy()

