
import numpy as np
from ..hubbard import HubbardInstance

def test_dimer():
    hi = HubbardInstance(L=2, N_up=1, N_down=1)
    hi.initialize()
    v = np.zeros(4)

    E_gnd, n_gnd = hi.generate_sample(v)
    print(E_gnd)
    print(n_gnd)


