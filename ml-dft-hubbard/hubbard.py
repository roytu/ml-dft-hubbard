
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

    def __init__(self, L=8, N_up=2, N_down=2):
        self.L      = L          # Number of sites
        self.N_up   = N_up       # Number of spin-up electrons
        self.N_down = N_down     # Number of spin-down electrons
        self.t = 1
        self.U = 4

        # Universal
        self.basis = None
        self.H_T = None
        self.H_U = None

    def initialize(self):
        # Generate basis vectors
        cs_up = HubbardInstance.__combinations_binary(self.L, self.N_up)
        cs_down = HubbardInstance.__combinations_binary(self.L, self.N_down)

        basis = []
        for c_up in cs_up:
            for c_down in cs_down:
                basis.append((c_up, c_down))
        self.basis = basis
        M = len(self.basis)

        # Construct H_T
        H_T = np.zeros((M, M), dtype=np.float64)
        #print("Constructing H_T")
        for i in range(M):
            (x_up, x_down) = self.basis[i]
            for j in range(M):
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
        #print(f"H_T = {H_T}")
        self.H_T = H_T

        # Construct H_U
        #print("Constructing H_U")
        h_u = np.zeros(M, dtype=np.float64)
        for i, (x_up, x_down) in enumerate(self.basis):
            # Get the number of sites with two electrons
            h_u[i] = self.U * sum([1 for b in bin(x_up & x_down)[2:] if b == "1"])
        H_U = np.diag(h_u)
        self.H_U = H_U

    def generate_sample(self, v):
        M = len(self.basis)

        # Construct H_V
        #print("Constructing H_V")
        h_v = np.zeros(M, dtype=np.float64)
        for i, (x_up, x_down) in enumerate(self.basis):
            # Convert integers into binary vectors
            b_up = [int(c) for c in str(bin(x_up)[2:]).rjust(self.L, "0")]
            v_up = np.dot(v, b_up)

            b_down = [int(c) for c in str(bin(x_down)[2:]).rjust(self.L, "0")]
            v_down = np.dot(v, b_down)

            h_v[i] = v_up + v_down
        H_V = np.diag(h_v)
        #print(f"H_V = {H_V}")

        # Add all hamiltonians
        H = self.H_T + self.H_U + H_V
        #print(f"H = {H}")

        # Compute ground state
        E_gnd, n_gnd = self.compute_ground_state(H)

        return E_gnd, n_gnd

    def compute_ground_state(self, H):
        """ Computes the ground state of the calculated hamiltonian.

            Ground state density is returned as a concatenation of the
            up and down spins: [n_gnd_up] + [n_gnd_down]

            Returns:
                (E_gnd, ground state density)
        """
        w, v = LA.eigh(H)
        E_gnd = w[0]
        p_gnd = v[0]
        n_gnd = p_gnd * p_gnd  # TODO: do we need to complex square this ever?

        # Convert p_gnd to n_gnd
        n_gnd_up = np.zeros(self.L, dtype=np.float64)
        n_gnd_down = np.zeros(self.L, dtype=np.float64)

        for i in range(self.L):
            b_up =   np.array([1 if (up   & (1 << (self.L - 1 - i))) > 0 else 0 for (up, down) in self.basis])
            b_down = np.array([1 if (down & (1 << (self.L - 1 - i))) > 0 else 0 for (up, down) in self.basis])

            # Convert occupation to density
            n_gnd_up[i]   = np.dot(n_gnd, b_up)
            n_gnd_down[i] = np.dot(n_gnd, b_down)

        n_gnd = np.hstack([n_gnd_up, n_gnd_down])

        #print(f"v = {v} -> E_gnd = {E_gnd}, n_gnd = {n_gnd}")

        return E_gnd, n_gnd

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
    # Create problem instance
    L = 8
    N_up = 2
    N_down = 2

    hi = HubbardInstance(L, N_up, N_down)
    hi.initialize()

    # Set a random potential
    W = 2.5  # Varies between 0.005t and 2.5t in the paper
    v = np.random.uniform(-W, W, L)

    # Solve ground state
    E_gnd, n_gnd = hi.generate_sample(v)

