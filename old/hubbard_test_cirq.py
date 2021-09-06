
import cirq

from recirq import fermi_hubbard
from recirq.fermi_hubbard import publication

# Hide numpy warnings
import warnings
warnings.filterwarnings("ignore")




"""Get all layouts for 8 sites on a 23-qubit subgrid of the Google Rainbow processor."""
layouts = publication.rainbow23_layouts(sites_count=8)
print(f"There are {len(layouts)} total qubit layouts.")

"""Get FermiHubbardParameters (problem descriptions) for each qubit layout with the above parameters."""
parameters = [
    publication.trapping_instance(layout, u=2, dt=0.3, up_particles=2, down_particles=2) 
    for layout in layouts
]

"""Display the Hamiltonian for an example problem description."""
parameters_example = parameters[0]
#print(parameters_example.hamiltonian)

#print(parameters_example.initial_state)
#print(parameters_example.dt)


"""Create circuits from a problem description."""
initial, trotter, measurement = fermi_hubbard.create_circuits(parameters_example, trotter_steps=1)

"""Display the total circuit to execute."""
circuit = initial + trotter + measurement
print(circuit)

"""Convert the circuit to native hardware gates perfectly (without noise)."""
publication.ideal_sqrt_iswap_converter().convert(circuit)

"""Set the number of Trotter steps to simulate."""
trotter_steps = range(10 + 1)

"""Get an ideal sampler to simulate experiments."""
ideal_sampler = fermi_hubbard.ConvertingSampler(
    cirq.Simulator(), publication.ideal_sqrt_iswap_converter().convert
)

"""Run the experiments on a perfect simulator for each qubit layout."""
from tqdm.notebook import tqdm

with tqdm(range(len(parameters) * len(trotter_steps))) as progress:
    experiments = [
        fermi_hubbard.run_experiment(
            params, 
            trotter_steps, 
            ideal_sampler, 
            post_run_func=lambda *_: progress.update()
        )
        for params in parameters
    ]
