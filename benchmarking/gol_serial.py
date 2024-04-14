from pylattica.core import SynchronousRunner
from pylattica.models.game_of_life import Life, GameOfLifeController
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup
from typing import List
import pandas as pd
import time

from pylattica.logger import setup_logger


logger = setup_logger("gol_serial_benchmark.log")


def simulate(size: List[int]):
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    starting_state = setup.setup_noise(
        size=size,
        phases=["dead", "alive"]
    )
    controller = GameOfLifeController(starting_state.structure, Life)
    runner = SynchronousRunner(parallel=False)
    runner.run(starting_state.state, controller, 60)


def benchmark_simulation(sizes: List[int], num_trials: int):
    execution_times = []
    for size in sizes:
        for trial in range(0, num_trials):

            logger.info(f"Beginning trial {trial} with size {size}")
            start_time = time.time()

            # Run the simulation
            simulate(size)

            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Completed trial {trial} with size {size}")

            execution_times.append((size, trial, execution_time))
    return execution_times


if __name__ == "__main__":
    sizes, num_trials = [10, 100], 3
    logger.info(f"Benchmarking simulation with sizes {sizes} runs for {num_trials} each\n")
    results = benchmark_simulation(sizes=sizes, num_trials=num_trials)
    df = pd.DataFrame(data=results, columns=["size", "trial", "time (seconds)"])
    print(df.to_string())
