
import numpy as np
from mldfthubbard.hubbard import HubbardInstance

def test_symmetric():
    """ Test that all Hamiltonians are symmetric
    """

    L = 8
    N_up = 2
    N_down = 2
    t = 1
    U = 1
    hi = HubbardInstance(L=L, N_up=N_up, N_down=N_down, t=t, U=U)
    hi.initialize()

    W = 0.005
    v = np.random.uniform(-W, W, L)

    hi.generate_sample(v)
    assert np.allclose(hi.H, hi.H.T)


