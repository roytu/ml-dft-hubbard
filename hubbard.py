
from itertools import combinations
import numpy as np
from numpy import linalg as LA

class HubbardInstance(object):
    """ Instance of a 1-D Hubbard model with periodic boundary conditions

        The Hamiltonian in question is:

        H = H_T + H_U + H_V

        where:

            H_T = -t (\Sum_{ijs} c^dag_{js} c_{is}  + h.c.)
            H_U = diag ( U \Sum_{is} n_{i up} n_{i down} )
            H_V = diag ( \Sum_{is} n_{is} v_i )

        (In H_T, i and j must be adjacent.)

        That is, the kinetic term, H_T, is a hopping Hamiltonian that allows for single-electron
        jumps to neighboring sites.  Note that spin is conserved.

        The potential term, H_U, models Coulomb repulsion.  It only contributes if two
        electrons share the same site (e.g. a spin-up and spin-down electron share the same site).

        The external potential, H_V, is a spin-independent v_i that penalizes electron density at
        certain sites.

        ( See: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.075132 )

        The basis is as follows (for L = 4, N_up = N_down = 2):

        ---------------------------
        | J | spin up | spin down |
        |--------------------------
        | 0 |    0011 |      0011 |
        |--------------------------
        | 1 |    0011 |      0101 |
        |--------------------------
        | 2 |    0011 |      0110 |
        |--------------------------
        | 3 |    0011 |      1001 |
        |--------------------------
        | 4 |    0011 |      1010 |
        |--------------------------
        |...|    ...  |      ...  |
        |--------------------------
        | M |    1100 |      1100 |
        |--------------------------

        That is, we specify each basis state as a lexicographical ordering of spin-up and spin-down occupation
        vectors, incrementing spin-down vectors first, then spin-up vectors once the spin-down vectors are exhausted.

        The dimensionality of the basis is:

            dim(J) = (L choose N_up) * (L choose N_down)

        For L = 8 and N_up = N_down = 2, dim(J) = 784
    """

    def __init__(self):
        self.L = 4          # Number of sites
        self.N_up = 2       # Number of spin-up electrons
        self.N_down = 2     # Number of spin-down electrons
        self.t = 1
        self.U = 4
        self.v = np.ones(self.L, dtype=np.float64)

        self.basis = None

        self.H_T = None
        self.H_U = None
        self.H_V = None

    def generate_basis(self):
        """ Generate basis vectors based on the ordering given above. """
        cs_up = HubbardInstance.__combinations_binary(self.L, self.N_up)
        cs_down = HubbardInstance.__combinations_binary(self.L, self.N_down)

        basis = []
        for c_up in cs_up:
            for c_down in cs_down:
                basis.append((c_up, c_down))
        self.basis = basis

    def construct_hamiltonians(self):

        self.generate_basis()
        M = len(self.basis)

        # Construct H_T
        H_T = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(M):
                (x_up, x_down) = self.basis[i]
                (y_up, y_down) = self.basis[j]

                if x_up == y_up:
                    # Check if x_down and y_down are 1 hop away from each other
                    if sum([1 for b in bin(x_down ^ y_down)[2:] if b == "1"]) == 2:
                        if self.N_up % 2 == 0:
                            H_T[i, j] = self.t
                        else:
                            H_T[i, j] = -self.t
                elif x_down == y_down:
                    # Check if x_up and y_up are 1 hop away from each other
                    if sum([1 for b in bin(x_up ^ y_up)[2:] if b == "1"]) == 2:
                        if self.N_down % 2 == 0:
                            H_T[i, j] = self.t
                        else:
                            H_T[i, j] = -self.t
        print(f"H_T = {H_T}")

        # Construct H_U
        h_u = np.zeros(M, dtype=np.float64)
        for i, (x_up, x_down) in enumerate(self.basis):
            # Get the number of sites with two electrons
            h_u[i] = self.U * sum([1 for b in bin(x_up & x_down)[2:] if b == "1"])

        H_U = np.diag(h_u)
        print(f"H_U = {H_U}")

        # Construct H_V
        h_v = np.zeros(M, dtype=np.float64)
        for i, (x_up, x_down) in enumerate(self.basis):
            # Convert integers into binary vectors
            b_up = [int(c) for c in str(bin(x_up)[2:]).rjust(self.L, "0")]
            v_up = np.dot(self.v, b_up)

            b_down = [int(c) for c in str(bin(x_down)[2:]).rjust(self.L, "0")]
            v_down = np.dot(self.v, b_down)

            h_v[i] = v_up + v_down
        H_V = np.diag(h_v)
        print(f"H_V = {H_V}")

        # Add all hamiltonians
        H = H_T + H_U + H_V
        print(f"H = {H}")

        return H

    @staticmethod
    def __combinations_binary(L, n):
        """ Return a list of integers whose binary representations are the 
            occupation vectors for 'L choose n', sorted.
        """
        xs = list(combinations(range(L), n))
        bin_xs = sorted([sum([1 << c for c in cs]) for cs in xs])
        return bin_xs

    def print_basis(self):
        def to_bin(x):
            return bin(x)[2:].rjust(self.L, "0")
        print([(to_bin(up), to_bin(down)) for (up, down) in self.basis])

if __name__ == "__main__":
    hi = HubbardInstance()
    H = hi.construct_hamiltonians()
    print()
    print(H)

    w, v = LA.eigh(H)
    E_gnd = w[0]
    p_gnd = v[0]
    n_gnd = p_gnd * p_gnd  # TODO: do we need to complex square this ever?
    print(f"E_gnd = {E_gnd}")
    print(f"p_gnd = {p_gnd}")

    # Convert p_gnd to n_gnd
    n_gnd_up = np.zeros(hi.L, dtype=np.float64)
    n_gnd_down = np.zeros(hi.L, dtype=np.float64)

    for i in range(hi.L):
        b_up =   np.array([1 if (up   & (1 << (hi.L - 1 - i))) > 0 else 0 for (up, down) in hi.basis])
        b_down = np.array([1 if (down & (1 << (hi.L - 1 - i))) > 0 else 0 for (up, down) in hi.basis])

        # Convert occupation to density
        n_gnd_up[i]   = np.dot(n_gnd, b_up   / (hi.N_up + hi.N_down))
        n_gnd_down[i] = np.dot(n_gnd, b_down / (hi.N_up + hi.N_down))

    n_gnd = np.hstack([n_gnd_up, n_gnd_down])
    print(n_gnd)
    print(np.sum(n_gnd))

    hi.print_basis()

