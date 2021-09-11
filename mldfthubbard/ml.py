
import time
import tensorflow as tf
from tensorflow import keras

import numpy as np
from hubbard import HubbardInstance

import matplotlib.pyplot as plt

def create_dataset():
    # Create problem instance
    L = 8
    N_up = 2
    N_down = 2

    hi = HubbardInstance(L=L, N_up=N_up, N_down=N_down, t=1, U=4)
    hi.initialize()

    SAMPLES = 100
    E_gnds = []
    n_gnds = []
    t = time.time()
    for i in range(SAMPLES):
        # Set a random potential
        W = 0.005  # Varies between 0.005t and 2.5t in the paper
        v = np.random.uniform(-W, W, L)
        v = np.zeros(L)
        E_gnd, n_gnd = hi.generate_sample(v)

        E_gnds.append(E_gnd)
        n_gnds.append(n_gnd)

        if i % 1000 == 0:
            dt = time.time() - t
            print(f"{i} / {SAMPLES}: {dt} s")
            t = time.time()

    E_gnds = np.array(E_gnds)
    n_gnds = np.array(n_gnds)

    np.save("E_gnds.npy", E_gnds)
    np.save("n_gnds.npy", n_gnds)

def load_dataset():
    E_gnds = np.load("E_gnds.npy")
    n_gnds = np.load("n_gnds.npy")

    print(E_gnds)
    print(n_gnds)

    plt.figure()

    E_ress = []
    n_ress = []

    for i in range(len(n_gnds)):
        N = len(n_gnds[i])
        hom = np.ones(N) * 1/4

        n_res = np.sqrt(np.sum((n_gnds[i] - hom) ** 2))

        E_ress.append(E_gnds[i])
        n_ress.append(n_res)

    plt.scatter(n_ress, E_ress)
    plt.show()


if __name__ == "__main__":
    create_dataset()
    load_dataset()

