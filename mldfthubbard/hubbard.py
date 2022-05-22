
import sys
from itertools import combinations
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import logging

from manim import *

from util import *

LOG_LEVEL = logging.WARN
logging.basicConfig(level=LOG_LEVEL)

# Don't truncate matrices
#np.set_printoptions(threshold=sys.maxsize)

class HubbardInstance(object):
    def __init__(self):
        self.t = 1
        self.U = 4
        self.sites = []
        self.links = []

        self.N_up = 0
        self.N_down = 0

        self.basis = []

        self.E_gnd = None
        self.n_gnd = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(LOG_LEVEL)

    def add_site(self):
        site = len(self.sites)
        self.sites.append(site)
        self.update_basis()
        return site

    def add_link(self, s1, s2):
        link = (s1, s2)
        self.update_basis()
        self.links.append(link)

    def update_basis(self):
        # Generate basis vectors
        L = len(self.sites)
        cs_up   = combinations_binary(L, self.N_up)
        cs_down = combinations_binary(L, self.N_down)

        basis = []
        for c_up in cs_up:
            for c_down in cs_down:
                basis.append((c_up, c_down))
        self.basis = basis
        M = len(self.basis)

    def calculate_ground_state(self):
        """ Computes the ground state of the calculated hamiltonian.

            Ground state density is returned as a concatenation of the
            up and down spins: [n_gnd_up] + [n_gnd_down]

            Returns:
                (E_gnd, ground state density)
        """
        w, v = LA.eigh(self.H)

        # Check for ground state degeneracy
        if np.abs(w[0] - w[1]) < 1e-10:
            self.logger.warn(f"GND state degeneracy: {w[0:5]}")

        E_gnd = w[0]
        self.logger.info(f"E_gnd = {E_gnd}")
        p_gnd = v[:, 0]
        self.logger.info(f"p_gnd = {p_gnd}")

        self.p_gnd = p_gnd  # Save for debugging purposes

        n_gnd = p_gnd ** 2  # TODO: do we need to complex square this ever?

        # Convert p_gnd to n_gnd
        L = len(self.sites)
        n_gnd_up = np.zeros(L, dtype=np.float64)
        n_gnd_down = np.zeros(L, dtype=np.float64)

        for i in range(L):
            b_up =   np.array([1 if self.is_occupied(i, up) else 0 for (up, down) in self.basis])
            b_down = np.array([1 if self.is_occupied(i, down) else 0 for (up, down) in self.basis])

            # Convert occupation to density
            n_gnd_up[i]   = np.dot(n_gnd, b_up)
            n_gnd_down[i] = np.dot(n_gnd, b_down)

        if not (np.abs(w[0] - w[1]) < 1e-10):
            assert np.allclose(n_gnd_up, n_gnd_down)
        n_gnd = np.hstack([n_gnd_up, n_gnd_down])

        self.logger.info(f"v = {v} -> E_gnd = {E_gnd}, n_gnd = {n_gnd}")

        self.E_gnd = E_gnd
        self.n_gnd = n_gnd

        # Compute expectation values
        self.exp_H   = self.p_gnd.T.dot(self.H).dot(self.p_gnd)
        self.exp_H_T = self.p_gnd.T.dot(self.H_T).dot(self.p_gnd)
        self.exp_H_U = self.p_gnd.T.dot(self.H_U).dot(self.p_gnd)
        self.exp_H_V = self.p_gnd.T.dot(self.H_V).dot(self.p_gnd)

        return E_gnd, n_gnd

    def calculate_Ht_and_Hu(self):
        self.H_T = self.calculate_Ht()
        self.H_U = self.calculate_Hu()
        return self.H_T, self.H_U

    def calculate_H(self):
        self.H = self.H_T + self.H_U + self.H_V
        return self.H

    def calculate_Ht(self):
        M = len(self.basis)
        H_T = np.zeros((M, M), dtype=np.float64)
        for (i, j) in self.links:
            for sp in ["up", "down"]:
                H_T += self.calculate_Ht_ij(i, j, sp)
                H_T += self.calculate_Ht_ij(j, i, sp)
        return H_T

    def calculate_Ht_ij(self, i, j, sp):
        # Generates an MxM matrix representing the operator (c_j^\dag c_i)
        # in the `self.basis` basis

        # Construct H_T
        M = len(self.basis)
        H_T_ij = np.zeros((M, M), dtype=np.float64)
        for k in range(M):
            (phi_k_up, phi_k_down) = self.basis[k]
            for l in range(M):
                (phi_l_up, phi_l_down) = self.basis[l]

                if sp == "up":
                    phi_k_same, phi_k_other = phi_k_up, phi_k_down
                    phi_l_same, phi_l_other = phi_l_up, phi_l_down
                elif sp == "down":
                    phi_k_same, phi_k_other = phi_k_down, phi_k_up
                    phi_l_same, phi_l_other = phi_l_down, phi_l_up

                # Verify that phi_k_other == phi_l_other
                if phi_k_other != phi_l_other:
                    H_T_ij[k, l] = 0
                    continue

                # Verify that phi_k_same has jth particle
                # Verify that phi_l_same has ith particle
                if (not self.is_occupied(j, phi_k_same)) or (not self.is_occupied(i, phi_l_same)):
                    H_T_ij[k, l] = 0
                    continue

                # Verify that besides i and j, phi_k_up == phi_l_up
                # (that is, we are only "one hop away")
                if not (self.add_particle_at(i, self.remove_particle_at(j, phi_k_same)) == phi_l_same):
                    H_T_ij[k, l] = 0
                    continue

                # We passed everything, aka this is a non-zero term

                # We need to calculate the sign of:
                # <y| c_l^\dag c_k |x>

                s = 1
                # Find how many hops it takes to get c_k to c_k^\dag in |k>
                u = len(self.sites) - 1
                while u > i:
                    if self.is_occupied(u, phi_k_same):
                        s *= -1
                    u -= 1

                # Find how many hops it takes to get c_l to c_l^\dag in |l>
                u = len(self.sites) - 1
                while u > j:
                    if self.is_occupied(u, phi_l_same):
                        s *= -1
                    u -= 1

                H_T_ij[k, l] = -self.t * s
                continue
        return H_T_ij

    def calculate_Hu(self):
        M = len(self.basis)
        h_U = np.zeros(M, dtype=np.float64)
        for k in range(M):
            (phi_k_up, phi_k_down) = self.basis[k]
            for i in self.sites:
                if self.is_occupied(i, phi_k_up) and self.is_occupied(i, phi_k_down):
                    h_U[k] += self.U
        return np.diag(h_U)

    def calculate_Hv(self):
        M = len(self.basis)
        H_V_i = np.zeros(M, dtype=np.float64)
        for i, (x_up, x_down) in enumerate(self.basis):
            s = 0
            for vi in range(len(self.v)):
                if self.is_occupied(vi, x_up):
                    s += self.v[vi]
                if self.is_occupied(vi, x_down):
                    s += self.v[vi]
            H_V_i[i] = s
        self.H_V = np.diag(H_V_i)
        return self.H_V

    def is_occupied(self, i, occupation_vector):
        return ((occupation_vector >> i) & 1) == 1

    def add_particle_at(self, i, phi):
        """ Add a particle to the i index at occupation vector `phi`
        """
        return (1 << i) | phi

    def remove_particle_at(self, i, phi):
        """ Remove a particle to the i index at occupation vector `phi`
        """
        return (~(1 << i)) & phi

    def graph(self):
        s = HubbardScene.create_from_instance(self)
        s.render()

    def print_basis(self):
        def to_bin(x):
            L = len(self.sites)
            return bin(x)[2:].rjust(L, "0")
        print([(to_bin(up), to_bin(down)) for (up, down) in self.basis])

    def print_calculations(self):
        print(f"sites = {self.sites}")
        print(f"links = {self.links}")
        print(f"basis = {self.basis}")
        print(f"H_T = {self.H_T}")
        print(f"H_U = {self.H_U}")
        print(f"H_V = {self.H_V}")
        print(f"H = {self.H}")
        print(f"E_gnd = {self.E_gnd}")
        print(f"n_gnd = {self.n_gnd}")

    @staticmethod
    def make_periodic_1d_chain(L):
        hi = HubbardInstance()
        for i in range(L):
            # Define lattice
            s1 = hi.add_site()

        for i in range(L):
            s1 = hi.sites[i]
            s2 = hi.sites[(i+1) % L]
            hi.add_link(s1, s2)

        return hi



class HubbardScene(Scene):
    def __init__(self):
        super().__init__()
        self.vertices = []
        self.edges = []
        
    @staticmethod
    def create_from_instance(hi):
        s = HubbardScene()
        s.vertices = hi.sites
        s.edges = hi.links
        return s

    def construct(self):
        g = Graph(
                self.vertices,
                self.edges,
                layout="circular",
                layout_scale=3,
                labels=True)
        self.add(g)


if __name__ == "__main__":
    # Create problem instance
    #L = 8
    #N_up = 2
    #N_down = 2

    #hi = HubbardInstance(L, N_up, N_down)
    #hi.initialize()

    ## Set a random potential
    #W = 2.5  # Varies between 0.005t and 2.5t in the paper
    #v = np.random.uniform(-W, W, L)

    ## Solve ground state
    #E_gnd, n_gnd = hi.generate_sample(v)

    # Initialize

    L = 8
    hi = HubbardInstance.make_periodic_1d_chain(L)

    # Add electrons
    hi.N_up = 2
    hi.N_down = 2

    # Graph
    hi.update_basis()
    hi.print_basis()

    hi.graph()

    # Calculate Ht, Hu
    H_T, H_U = hi.calculate_Ht_and_Hu()

    # Set a random potential
    W = 2.5  # Varies between 0.005t and 2.5t in the paper
    v = np.random.uniform(-W, W, L)
    hi.v = v

    # Calculate Hv
    H_v = hi.calculate_Hv()

    # Calculate H
    H = hi.calculate_H()

    # Solve ground state
    E_gnd, n_gnd = hi.calculate_ground_state()

    hi.print_calculations()

    # Setup
    fig, ax1 = plt.subplots()

    ax1.set_title(f"Random external potential, n (W={W})")
    ax1.set_xlabel("Site")
    ax2 = ax1.twinx()
    ax1.set_ylabel("n_gnd")
    ax2.set_ylabel("v")
    ax1.tick_params(axis='y', colors='black')
    ax2.tick_params(axis='y', colors='red')

    ax1.plot(n_gnd[:int(len(n_gnd)/2)], color="black", label="n")
    ax2.plot(v, color="red", label="v")

    plt.legend(loc="best")
    plt.show()

    print(E_gnd)
    print(n_gnd)
    print(v)
