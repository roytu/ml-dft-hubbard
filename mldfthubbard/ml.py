
import time
import tensorflow as tf
from tensorflow import keras

import numpy as np
from hubbard import HubbardInstance

import matplotlib.pyplot as plt


class Results(object):
    def __init__(self):
        self.L = None
        self.t = None
        self.U = None
        self.E_gnds = None
        self.n_gnds = None
        self.vs = None
        self.exp_Hs = None
        self.exp_H_Ts = None
        self.exp_H_Us = None
        self.exp_H_Vs = None

def run_experiment(SAMPLES, L=8, N_up=2, N_down=2, t=1, U=4, W=2.5, set_v=None):
    """ Generates results for `SAMPLES` iterations.

    Args:
        L: number of lattice sites
        N_up, N_down: number of spin-up and spin-down electrons
        t: hopping parameter
        U: Coulomb potential paremter

    Returns:
        Results object
    """

    # Create problem instance
    hi = HubbardInstance(L=L, N_up=N_up, N_down=N_down, t=t, U=U)
    hi.initialize()

    E_gnds = []
    n_gnds = []
    vs = []

    exp_Hs = []
    exp_H_Ts = []
    exp_H_Us = []
    exp_H_Vs = []

    it = time.time()
    for i in range(SAMPLES):
        # Set a random potential
        #W = 0.005  # Varies between 0.005t and 2.5t in the paper
        #W = 2.5  # Varies between 0.005t and 2.5t in the paper
        #W = 1000000  # Varies between 0.005t and 2.5t in the paper
        if set_v is not None:
            v = set_v
        else:
            v = np.random.uniform(-W, W, L)
            if i == 0:
                # First sample is always homogenous
                v = np.zeros(L)

        # Subtract out bias
        v -= np.sum(v) / L

        E_gnd, n_gnd = hi.generate_sample(v)

        E_gnds.append(E_gnd)
        n_gnds.append(n_gnd)
        vs.append(v)
        exp_Hs.append(hi.exp_H)
        exp_H_Ts.append(hi.exp_H_T)
        exp_H_Us.append(hi.exp_H_U)
        exp_H_Vs.append(hi.exp_H_V)

        # Print progress every 1000th iteration
        if i % 1000 == 0 and i > 0:
            dt = time.time() - it
            print(f"{i} / {SAMPLES}: {dt} s")
            it = time.time()

    # Save to CSV
    #with open("results.csv", "w") as f:
    #    f.write("|".join([
    #        "E_gnd", "n_gnd", "v", "exp_H", "exp_H_T", "exp_H_U", "exp_H_V"
    #        ]) + "\n")
    #    for i in range(SAMPLES):
    #        xs = []

    #        xs.append(str(E_gnds[i]))
    #        xs.append(",".join([str(x) for x in n_gnds[i]]))
    #        xs.append(",".join([str(x) for x in vs[i]]))
    #        xs.append(str(exp_Hs[i]))
    #        xs.append(str(exp_H_Ts[i]))
    #        xs.append(str(exp_H_Us[i]))
    #        xs.append(str(exp_H_Vs[i]))

    #        f.write("|".join(xs) + "\n")

    # Assemble results
    E_gnds   = np.array(E_gnds)
    n_gnds   = np.array(n_gnds)
    vs       = np.array(vs)
    exp_Hs   = np.array(exp_Hs)
    exp_H_Ts = np.array(exp_H_Ts)
    exp_H_Us = np.array(exp_H_Us)
    exp_H_Vs = np.array(exp_H_Vs)

    results = Results()
    results.L = L
    results.t = t
    results.U = U
    results.E_gnds = E_gnds
    results.n_gnds = n_gnds
    results.vs = vs
    results.exp_Hs = exp_Hs
    results.exp_H_Ts = exp_H_Ts
    results.exp_H_Us = exp_H_Us
    results.exp_H_Vs = exp_H_Vs

    return results


def load_dataset(Ws):
    # Create table
    #for W in Ws:
    #    E_gnds = np.load(f"E_gnds_{W}.npy", E_gnds)
    #    n_gnds = np.load(f"n_gnds_{W}.npy", n_gnds)
    #    vs = np.load(f"vs_{W}.npy", vs)


    ## Plot all ns
    #plt.figure()
    #plt.ylabel("$n$")
    #for i in range(len(vs)):

    #    plt.plot(n_gnds[i][0:8])
    #plt.show()

    # Plot vs / n
    for i in range(len(vs)):

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Site")

        # vs
        ax1.set_ylabel("$vs$")
        ax1.plot(vs[i], label="vs", color="tab:red")

        # ns
        ax2 = ax1.twinx()
        ax2.set_ylabel("ns")
        ax2.plot(n_gnds[i][0:8], label="ns", color="tab:blue")

        fig.tight_layout()
        plt.show()

    #np.save(f"E_gnds_{W}.npy", E_gnds)
    #np.save(f"n_gnds_{W}.npy", n_gnds)
    #np.save(f"vs_{W}.npy", vs)

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

def gen_increasing_t():
    """ Random potential, increasing T. """

    L = 8
    W = 2.5
    t = 1
    U = 4

    v = np.random.uniform(-W, W, L)
    v -= np.sum(v) / L

    # Setup
    fig, ax1 = plt.subplots()

    ax1.set_title("Random external potential, various t values")
    ax2 = ax1.twinx()
    ax1.set_ylabel("n_gnd")

    ax2.plot(v, color="black", label="v")
    ax2.set_ylabel("v")

    # Execute
    ts = [1, 2, 5, 10, 100, 1000]

    for t in ts:
        results = run_experiment(1, L=L, t=t, U=U, set_v=v)

        n_gnd = results.n_gnds[0]
        
        ax1.plot(n_gnd[:int(len(n_gnd)/2)], label=f"t = {t}")

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    plt.show()

def gen_increasing_U():
    """ Random potential, increasing U. """

    L = 8
    W = 2.5
    t = 1
    U = 4

    v = np.random.uniform(-W, W, L)
    v -= np.sum(v) / L

    # Setup
    fig, ax1 = plt.subplots()

    ax1.set_title("Random external potential, various U values")
    ax2 = ax1.twinx()
    ax1.set_ylabel("n_gnd")

    ax2.plot(v, color="black", label="v")
    ax2.set_ylabel("v")

    # Execute
    Us = [0, 1, 2, 5, 10, 100, 1000]

    for U in Us:
        results = run_experiment(1, L=L, t=t, U=U, set_v=v)

        n_gnd = results.n_gnds[0]
        
        ax1.plot(n_gnd[:int(len(n_gnd)/2)], label=f"U = {U}")

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    plt.show()

def gen_random_v_and_n():
    """ Random potential, resulting n. """

    SAMPLES = 3

    L = 8
    W = 2.5
    t = 1
    U = 4

    # Execute
    results = run_experiment(SAMPLES + 1, L=L, t=t, U=U, W=W)

    for i in range(1, SAMPLES + 1):
        # Setup
        fig, ax1 = plt.subplots()

        ax1.set_title(f"Random external potential, n (W={W})")
        ax1.set_xlabel("Site")
        ax2 = ax1.twinx()
        ax1.set_ylabel("n_gnd")
        ax2.set_ylabel("v")
        ax1.tick_params(axis='y', colors='black')
        ax2.tick_params(axis='y', colors='red')

        n_gnd = results.n_gnds[i]
            
        ax1.plot(n_gnd[:int(len(n_gnd)/2)], color="black", label="n")
        ax2.plot(results.vs[i], color="red", label="v")

        plt.show()


if __name__ == "__main__":
    #gen_increasing_t()
    #gen_increasing_U()
    gen_random_v_and_n()
    #Ws = [0.005, 0.010, 0.1, 1, 2.5]
    #create_dataset(2.5)
    #for W in Ws:
    #    create_dataset(W)
    #load_dataset(Ws)

