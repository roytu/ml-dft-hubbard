
import numpy as np
from mldfthubbard.hubbard import HubbardInstance

def test_dimer():
    """ Testing Hubbard dimer (L=2)

    Ground state of Hubbard dimer with no external potential is

        E_0 = U / 2 - W

    where

        W = sqrt((U/2) ** 2 + 4t ** 2)

    Ref: https://www.cond-mat.de/events/correl17/manuscripts/eder.pdf (pg. 6)
    """

    # Create Hubbard dimer with no external potential
    t = 1
    U = 1
    hi = HubbardInstance(L=2, N_up=1, N_down=1, t=t, U=U)
    hi.initialize()
    v = np.zeros(2)

    E_gnd, n_gnd = hi.generate_sample(v)

    E_gnd_true = U/2 - np.sqrt((U/2) ** 2 + 4 * t ** 2)
    n_gnd_true = 0.5 * np.ones(4)

    assert np.abs(E_gnd - E_gnd_true) < 1e-5
    assert np.sum(np.abs(n_gnd - n_gnd_true)) < 1e-5

    print()
    print(f"E_gnd: {E_gnd}")
    print(f"n_gnd: {n_gnd}")

