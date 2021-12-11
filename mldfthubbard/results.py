
import os
import time

from datetime import datetime
import numpy as np


class Results(object):
    def __init__(self):
        self.SAMPLES = None
        self.L = None
        self.W = None
        self.t = None
        self.U = None
        self.E_gnds = None
        self.n_gnds = None
        self.vs = None
        self.exp_Hs = None
        self.exp_H_Ts = None
        self.exp_H_Us = None
        self.exp_H_Vs = None

    def save(self, name=None):
        """ Saves the results in results/<DIRECTORY>.

            Returns the name (useful for auto-generated results).
        """

        if name is None:
            now = datetime.now()
            name = now.strftime("%Y%m%d_%H%M%S")

        direc = f"results/{name}"
        os.makedirs(direc, exist_ok=True)

        # Save npy files
        def _verbose_save(filename, values):
            print(f"Writing to {filename}...")
            np.save(f"{filename}", values)

        _verbose_save(f"{direc}/SAMPLES.npy", self.SAMPLES)
        _verbose_save(f"{direc}/L.npy", self.L)
        _verbose_save(f"{direc}/W.npy", self.W)
        _verbose_save(f"{direc}/t.npy", self.t)
        _verbose_save(f"{direc}/U.npy", self.U)
        _verbose_save(f"{direc}/E_gnds.npy", self.E_gnds)
        _verbose_save(f"{direc}/n_gnds.npy", self.n_gnds)
        _verbose_save(f"{direc}/vs.npy", self.vs)
        _verbose_save(f"{direc}/exp_Hs.npy", self.exp_Hs)
        _verbose_save(f"{direc}/exp_H_Ts.npy", self.exp_H_Ts)
        _verbose_save(f"{direc}/exp_H_Us.npy", self.exp_H_Us)
        _verbose_save(f"{direc}/exp_H_Vs.npy", self.exp_H_Vs)

        return name

    def load(self, name):
        """ Loads the results in results/<DIRECTORY>.
        """

        direc = f"results/{name}"

        # Save npy files
        print(f"Loading from {direc}...")

        self.SAMPLES = np.load(f"{direc}/SAMPLES.npy")
        self.L = np.load(f"{direc}/L.npy")
        self.W = np.load(f"{direc}/W.npy")
        self.t = np.load(f"{direc}/t.npy")
        self.U = np.load(f"{direc}/U.npy")
        self.E_gnds = np.load(f"{direc}/E_gnds.npy")
        self.n_gnds = np.load(f"{direc}/n_gnds.npy")
        self.vs = np.load(f"{direc}/vs.npy")
        self.exp_Hs = np.load(f"{direc}/exp_Hs.npy")
        self.exp_H_Ts = np.load(f"{direc}/exp_H_Ts.npy")
        self.exp_H_Us = np.load(f"{direc}/exp_H_Us.npy")
        self.exp_H_Vs = np.load(f"{direc}/exp_H_Vs.npy")

    def show(self):
        """ Prints information.
        """

        s = ""
        s += f"SAMPLES = {self.SAMPLES}\n"
        s += f"L = {self.L}\n"
        s += f"W = {self.W}\n"
        s += f"t = {self.t}\n"
        s += f"U = {self.U}\n"

        print(s)


    def plot_vs_over_n(self):
        # Plot vs / n
        for i in range(len(vs)):

            fig, ax1 = plt.subplots()
            ax1.set_xlabel("Site")

            # vs
            ax1.set_ylabel("$vs$")
            ax1.plot(self.vs[i], label="vs", color="tab:red")

            # ns
            ax2 = ax1.twinx()
            ax2.set_ylabel("ns")
            ax2.plot(self.n_gnds[i][0:8], label="ns", color="tab:blue")

            fig.tight_layout()
            plt.show()


    def plot_all_ns(self):
        # Plot all ns
        plt.figure()
        plt.ylabel("$n$")
        for i in range(len(vs)):
            plt.plot(self.n_gnds[i][0:8])
        plt.show()


    def plot_n_vs_E(self):
        # Plot n/E
        plt.figure()
        plt.xlabel("$|n - n_{homo}|$")
        plt.ylabel("$(E - E_{homo})$")

        E_ress = []
        n_ress = []

        # First sample is always E_homo / n_homo
        E_homo = self.E_gnds[0]
        #n_homo = self.n_gnds[0]

        L = 8
        N = 4
        n_homo = np.ones(L * 2) / N  # Force this because n_gnd for degenerate ground state spaces is poorly defined

        for i in range(len(self.n_gnds)):
            # Calculate n residual
            n_res = np.sqrt(np.sum((self.n_gnds[i] - n_homo) ** 2))

            # Calculate E residual
            E_res = (self.E_gnds[i] - E_homo)

            # Append results
            E_ress.append(E_res)
            n_ress.append(n_res)

        plt.scatter(n_ress, E_ress, label=W)
        plt.legend()
        plt.show()



