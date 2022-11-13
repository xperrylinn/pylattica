from pylattica.core import Runner, Simulation
from pylattica.discrete import PhaseSet
from .grid_setup import DiscreteGridSetup
from ..models.growth import GrowthController

# from ..square_grid.neighborhoods import PseudoHexagonalNeighborhoodBuilder2D

class GrowthSetup():

    def __init__(self, phase_set: PhaseSet):
        self._phases = phase_set

    def grow(self, size, num_sites_desired: int, background_spec, nuc_species, nb_builder, nuc_ratios = None, buffer = 2):
        setup = DiscreteGridSetup(self._phases)
        simulation = setup.setup_random_sites(
            size,
            num_sites_desired=num_sites_desired,
            background_spec=background_spec,
            nuc_species=nuc_species,
            nuc_ratios=nuc_ratios,
            buffer=buffer
        )

        controller = GrowthController(
            self._phases,
            simulation.structure,
            nb_builder=nb_builder,
            background_phase=background_spec
        )

        runner = Runner(parallel=True)
        res = runner.run(simulation.state, controller, num_steps = size)
        return Simulation(res.last_step, simulation.structure)