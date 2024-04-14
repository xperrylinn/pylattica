from pylattica.core import SynchronousRunner
from pylattica.models.game_of_life import Life, GameOfLifeController
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup
from pylattica.logger import logger
from typing import List, Callable
import pandas as pd
import time

# Constants
gol_phases = ["dead", "alive"]
num_steps = 100
sizes = [10, 100, 1000]
sizes = [10]
num_trials = 3


def simulate_serial(size: List[int], num_steps: int):
    phases = PhaseSet(phases=gol_phases)
    setup = DiscreteGridSetup(phase_set=phases)
    starting_state = setup.setup_noise(
        size=size,
        phases=gol_phases
    )
    controller = GameOfLifeController(starting_state.structure, Life)
    runner = SynchronousRunner(parallel=False)
    runner.run(starting_state.state, controller, num_steps=num_steps)


def simulate_parallel(size: List[int], num_steps: int):
    phases = PhaseSet(phases=gol_phases)
    setup = DiscreteGridSetup(phase_set=phases)
    starting_state = setup.setup_noise(
        size=size,
        phases=gol_phases
    )
    controller = GameOfLifeController(starting_state.structure, Life)
    runner = SynchronousRunner(parallel=False)
    runner.run(starting_state.state, controller, num_steps=num_steps)


def benchmark_simulation(fxn: Callable[[int, int], None], sizes: List[int], num_trials: int, num_steps: int):
    execution_times = []
    for size in sizes:
        for trial in range(0, num_trials):

            logger.info(f"Beginning trial {trial} with size {size}")
            start_time = time.time()

            # Run the simulation
            fxn(size, num_steps)

            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Completed trial {trial} with size {size}")

            execution_times.append((size, trial, num_steps, execution_time))
    return execution_times


if __name__ == "__main__":
    logger.info(f"Benchmarking simulation with sizes {sizes} runs for {num_trials} each\n")
    results = benchmark_simulation(fxn=simulate_serial, sizes=sizes, num_trials=num_trials, num_steps=num_steps)
    logger.info("Finished collecting results. Writing to CSV.")
    df = pd.DataFrame(data=results, columns=["size", "trial", "num_steps", "time (seconds)"])
    output_file_path = "gol_serial.csv"
    df.to_csv(path_or_buf=output_file_path, index=False)
