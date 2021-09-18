
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
    t = 1
    U = 4

    hi = HubbardInstance(L=L, N_up=N_up, N_down=N_down, t=t, U=U)
    hi.initialize()

    SAMPLES = 10
    E_gnds = []
    n_gnds = []
    it = time.time()
    for i in range(SAMPLES):
        # Set a random potential
        W = 0.005  # Varies between 0.005t and 2.5t in the paper
        v = np.random.uniform(-W, W, L)
        if i == 0:
            # First sample is always homogenous
            v = np.zeros(L)
        E_gnd, n_gnd = hi.generate_sample(v)

        E_gnds.append(E_gnd)
        n_gnds.append(n_gnd)

        # Print progress every 1000th iteration
        if i % 1000 == 0 and i > 0:
            dt = time.time() - it
            print(f"{i} / {SAMPLES}: {dt} s")
            it = time.time()

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
    plt.xlabel("$|n - n_{homo}|$")
    plt.ylabel("$(E - E_{homo})$")

    E_ress = []
    n_ress = []

    # First sample is always E_homo / n_homo
    E_homo = E_gnds[0]
    #n_homo = n_gnds[0]
    n_homo = np.ones(len(n_gnds[0])) * 1/4  # Force this because n_gnd for degenerate ground state spaces is poorly defined

    for i in range(len(n_gnds)):
        print(np.sum(n_gnds[i]))
        # Calculate n residual
        n_res = np.sqrt(np.sum((n_gnds[i] - n_homo) ** 2))

        # Calculate E residual
        E_res = (E_gnds[i] - E_homo)

        # Append results
        E_ress.append(E_res)
        n_ress.append(n_res)

    plt.scatter(n_ress, E_ress)
    plt.show()


if __name__ == "__main__":
    create_dataset()
    load_dataset()
