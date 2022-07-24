#/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

OFF, START, FINISH = -1, 1, 2

def racetrack_0():
    arr = np.zeros((32, 16), dtype=int)
    arr[0, :3] = OFF
    arr[1:3, :2] = OFF
    arr[3, 0] = OFF
    arr[14:22, 0] = OFF
    arr[22:29, :2] = OFF
    arr[29:, :3] = OFF

    arr[6, 9:] = OFF
    arr[7:, 8:] = OFF

    arr[-1, 3:8] = START
    arr[0:6, -1] = FINISH
    return arr

def racetrack_1():
    arr = np.zeros((30, 32), dtype=int)
    arr[0, :16] = OFF
    arr[1, :13] = OFF
    arr[2, :12] = OFF 
    arr[3:7, :11] = OFF
    arr[7, :12] = OFF
    arr[8, :13] = OFF
    arr[9:14, :14] = OFF
    for i in range(14):
        arr[i + 14, :13 - i] = OFF

    arr[9, 30:] = OFF
    arr[10, 27:] = OFF
    arr[11, 26:] = OFF
    arr[12, 25:] = OFF
    arr[13, 23:] = OFF
    arr[14:, 22:] = OFF

    arr[-1:, :22] = START
    arr[:9, -1] = FINISH
    return arr

RACETRACK, FNAME = racetrack_1(), "racetrack_1"
ON_TRACK = np.array(np.where(RACETRACK != OFF), dtype=int).swapaxes(0, 1)
OFF_TRACK = np.array(np.where(RACETRACK == OFF), dtype=int).swapaxes(0, 1)
ON_FINISH = np.array(np.where(RACETRACK == FINISH), dtype=int).swapaxes(0, 1)
ON_START = np.array(np.where(RACETRACK == START), dtype=int).swapaxes(0, 1)

def main():
    sns.heatmap(RACETRACK, cbar=False, xticklabels=False,
                yticklabels=False, square=True)
    plt.savefig(FNAME + '.svg')

if __name__ == '__main__':
    main()
