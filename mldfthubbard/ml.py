
import time
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

try:
    from tqdm.notebook import trange, tqdm
except ImportError:
    tqdm = None

from .hubbard import HubbardInstance
from .results import Results


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

    # Initialize tqdm if it exists

    if tqdm is None:
        pbar = None
    else:
        pbar = tqdm.tqdm(total=SAMPLES)

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

        if pbar:
            pbar.update(i)
        else:
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
    results.SAMPLES = SAMPLES
    results.L = L
    results.W = W
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


if __name__ == "__main__":
    #Ws = [0.005, 0.010, 0.1, 1, 2.5]
    res = run_experiment(10, W=0.005)
    name = res.save()

    # Load
    nres = Results()
    nres.load(name)
    nres.show()

