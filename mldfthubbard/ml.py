
import time
import tensorflow as tf
from tensorflow import keras

import numpy as np
from hubbard import HubbardInstance

import matplotlib.pyplot as plt

def create_dataset(W=2.5):
    # Create problem instance
    L = 8
    N_up = 2
    N_down = 2
    t = 1
    U = 4

    hi = HubbardInstance(L=L, N_up=N_up, N_down=N_down, t=t, U=U)
    hi.initialize()

    SAMPLES = 100
    E_gnds = []
    n_gnds = []
    vs = []
    it = time.time()
    for i in range(SAMPLES):
        # Set a random potential
        #W = 0.005  # Varies between 0.005t and 2.5t in the paper
        #W = 2.5  # Varies between 0.005t and 2.5t in the paper
        #W = 1000000  # Varies between 0.005t and 2.5t in the paper
        v = np.random.uniform(-W, W, L)

        # Subtract out bias
        v -= np.sum(v) / L

        if i == 0:
            # First sample is always homogenous
            v = np.zeros(L)
        E_gnd, n_gnd = hi.generate_sample(v)

        E_gnds.append(E_gnd)
        n_gnds.append(n_gnd)
        vs.append(v)

        # Print progress every 1000th iteration
        if i % 1 == 0 and i > 0:
            dt = time.time() - it
            print(f"{i} / {SAMPLES}: {dt} s")
            it = time.time()

    E_gnds = np.array(E_gnds)
    n_gnds = np.array(n_gnds)
    vs = np.array(vs)

    #np.save("npys/W_2.5_N_52500_004/E_gnds.npy", E_gnds)
    #np.save("npys/W_2.5_N_52500_004/n_gnds.npy", n_gnds)

    np.save(f"E_gnds_{W}.npy", E_gnds)
    np.save(f"n_gnds_{W}.npy", n_gnds)
    #np.save("vs.npy", vs)

def load_dataset(Ws):
    #E_gnds = np.load("E_gnds.npy")
    #n_gnds = np.load("n_gnds.npy")
    #vs = np.load("vs.npy")

    # Plot all ns
    #plt.figure()
    #plt.ylabel("$n$")
    #for i in range(len(vs)):

    #    plt.plot(n_gnds[i][0:8])
    #plt.show()

    # Plot vs / n
    #for i in range(len(vs)):

    #    fig, ax1 = plt.subplots()
    #    ax1.set_xlabel("Site")

    #    # vs
    #    ax1.set_ylabel("$vs$")
    #    ax1.plot(vs[i], label="vs", color="tab:red")

    #    # ns
    #    ax2 = ax1.twinx()
    #    ax2.set_ylabel("ns")
    #    ax2.plot(n_gnds[i][0:8], label="ns", color="tab:blue")

    #    fig.tight_layout()
    #    plt.show()


    ## Plot n/E
    plt.figure()
    plt.xlabel("$|n - n_{homo}|$")
    plt.ylabel("$(E - E_{homo})$")

    for W in Ws:
        E_gnds = np.load(f"E_gnds_{W}.npy")
        n_gnds = np.load(f"n_gnds_{W}.npy")
        #vs = np.load("vs.npy")

        E_ress = []
        n_ress = []

        # First sample is always E_homo / n_homo
        E_homo = E_gnds[0]
        #n_homo = n_gnds[0]

        L = 8
        N = 4
        n_homo = np.ones(L * 2) / N  # Force this because n_gnd for degenerate ground state spaces is poorly defined

        for i in range(len(n_gnds)):
            # Calculate n residual
            n_res = np.sqrt(np.sum((n_gnds[i] - n_homo) ** 2))

            # Calculate E residual
            E_res = (E_gnds[i] - E_homo)

            # Append results
            E_ress.append(E_res)
            n_ress.append(n_res)

        plt.scatter(n_ress, E_ress, label=W)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    Ws = [0.01, 0.100, 1, 2, 2.5]
    for W in Ws:
        create_dataset(W)
    load_dataset(Ws)

