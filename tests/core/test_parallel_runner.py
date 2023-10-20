import pytest

from pylattica.core import Runner, BasicController
from pylattica.core.simulation_state import SimulationState
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.constants import SITE_ID

import time

def test_parallel_runner(square_grid_2D_4x4: PeriodicStructure):
    
    class SimpleParallelController(BasicController):

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            prev = prev_state.get_site_state(site_id)["value"]
            new_state = prev + 1
            return {
                site_id: {
                    "value": new_state
                }
            }
    
    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], { "value": 0 })
    
    runner = Runner(parallel=True)

    controller = SimpleParallelController(square_grid_2D_4x4)
    result = runner.run(initial_state, controller=controller, num_steps = 1000, structure = square_grid_2D_4x4)

    last_step = result.last_step

    for site_state in last_step.all_site_states():
        assert site_state["value"] == 1000

def test_parallel_runner_speed(square_grid_2D_4x4: PeriodicStructure):
    
    class SimpleParallelController(BasicController):

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            prev = prev_state.get_site_state(site_id)["value"]
            new_state = prev + 1
            return {
                site_id: {
                    "value": new_state
                }
            }
    
    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], { "value": 0 })
    

    
    parallel_runner = Runner(parallel=True)
    series_runner = Runner()

    controller = SimpleParallelController(square_grid_2D_4x4)

    num_steps = 1000

    t0 = time.time()
    parallel_result = parallel_runner.run(initial_state, controller=controller, num_steps = num_steps, structure = square_grid_2D_4x4)
    t1 = time.time()

    t2 = time.time()
    series_result = series_runner.run(initial_state, controller=controller, num_steps = num_steps, structure = square_grid_2D_4x4)
    t3 = time.time()

    assert (t3 - t2) < (t1 - t0)

    for site_state in parallel_result.last_step.all_site_states():
        assert site_state["value"] == num_steps

    for site_state in series_result.last_step.all_site_states():
        assert site_state["value"] == num_steps