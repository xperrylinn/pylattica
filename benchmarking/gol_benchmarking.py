from pylattica.core import SynchronousRunner
from pylattica.models.game_of_life import Life, GameOfLifeController
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup
from pylattica.logger import logger
from typing import List, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    logger.info(f"Benchmarking simulation with sizes {sizes} runs for {num_trials} and num_steps {num_steps} each")

    logger.info(f"Benchmarking {simulate_serial.__name__}")
    results = benchmark_simulation(fxn=simulate_serial, sizes=sizes, num_trials=num_trials, num_steps=num_steps)
    logger.info(f"Finished collecting {simulate_serial.__name__} results. Writing to CSV.")
    serial_df = pd.DataFrame(data=results, columns=["size", "trial", "num_steps", "time (seconds)"])
    output_file_path = "gol_serial.csv"
    serial_df.to_csv(path_or_buf=output_file_path, index=False)

    logger.info(f"Benchmarking {simulate_parallel.__name__}")
    results = benchmark_simulation(fxn=simulate_parallel, sizes=sizes, num_trials=num_trials, num_steps=num_steps)
    logger.info(f"Finished collecting {simulate_parallel.__name__} results. Writing to CSV.")
    parallel_df = pd.DataFrame(data=results, columns=["size", "trial", "num_steps", "time (seconds)"])
    output_file_path = "gol_parallel.csv"
    parallel_df.to_csv(path_or_buf=output_file_path, index=False)

    logger.info("Data aggregation")
    serial_df = serial_df.groupby(by=["size", "num_steps"]).agg({"time (seconds)": np.mean})
    parallel_df = parallel_df.groupby(by=["size", "num_steps"]).agg({"time (seconds)": np.mean})

    logger.info("Plotting")
    plt.figure(figsize=(10, 6))

    plt.plot(sizes, serial_df["time (seconds)"], label="serial", color='blue', linestyle='-', linewidth=2)
    plt.plot(sizes, parallel_df["time (seconds)"], label="parallel", color='red', linestyle='--', linewidth=2)

    plt.xlabel("size")
    plt.ylabel("time (seconds)")
    plt.title("Original Implementation\nParallel & Serial Run Times")

    plt.legend(loc="upper right")

    output_file = "parallel_serial_original.png"
    plt.savefig(output_file)

    plt.grid(True)
 
    logger.info(f"Plot saved as '{output_file}'")

