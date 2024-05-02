from pylattica.core import SynchronousRunner
from pylattica.models.game_of_life import Life, GameOfLifeController
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup


phases = PhaseSet(["dead", "alive"])
setup = DiscreteGridSetup(phases)
starting_state = setup.setup_noise(
    size=4,
    phases=["dead", "alive"]
)
controller = GameOfLifeController(starting_state.structure, Life)
runner = SynchronousRunner(parallel=False)
runner.run(starting_state.state, controller, 60)

