import pytest

from pylattica.core import SynchronousRunner
from pylattica.models.game_of_life import Life
from pylattica.discrete import PhaseSet, DiscreteResultAnalyzer
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup

@pytest.fixture()
def discrete_result():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_noise(10, ["dead", "alive"])
    controller = Life(structure = simulation.structure)
    runner = SynchronousRunner(parallel=True)
    return runner.run(simulation.state, controller, 10, verbose=False)

def test_plot_phase_fractions(discrete_result):
    analyzer = DiscreteResultAnalyzer(discrete_result)
    analyzer.plot_phase_fractions()

def test_final_phase_fractions(discrete_result):
    analyzer = DiscreteResultAnalyzer(discrete_result)
    fracs = analyzer.final_phase_fractions()
    for phase, amt in fracs.items():
        assert type(phase) is str
        assert type(amt) is float

def test_phase_fraction_at_step():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_interface(10, "dead", "alive")
    controller = Life(structure = simulation.structure)
    runner = SynchronousRunner(parallel=True)
    result = runner.run(simulation.state, controller, 10, verbose=False)

    analyzer = DiscreteResultAnalyzer(result)

    assert analyzer.phase_fraction_at(0, "dead") == 0.5

