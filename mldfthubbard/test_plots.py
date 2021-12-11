""" Generate test plots """

import numpy as np
import matplotlib.pyplot as plt

from .hubbard import HubbardInstance

def gen_increasing_t():
    """ Generate a plot with a random external potential and 
        increasing values of t.
    """

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
    """ Generate a plot with a random external potential and 
        increasing values of U.
    """

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
    """ Generate a plot with a random external potential and 
        the resulting probability density.
    """

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
    #gen_random_v_and_n()
    pass
