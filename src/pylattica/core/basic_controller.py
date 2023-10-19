from abc import ABC, abstractmethod
import random

from .simulation_result import SimulationResult
from .simulation_state import SimulationState
from .periodic_structure import PeriodicStructure


class BasicController(ABC):
    """The base class which all Controllers extend. Every new type of
    simulation will involve creating a new Controller class. The controller
    class has a single responsibility, which is to implement the update
    rule for the simulation.

    To do this, implement the get_state_update method. The entire current
    SimulationState will be passed to this method, along with the ID of
    the site at which the update rule should be applied. It is up to the
    user to decide what updates should be produced using this information.
    """

    is_async = False

    def __init__(self, struct: PeriodicStructure):
        self.structure = struct

    @abstractmethod
    def get_state_update(self, site_id: int, prev_state: SimulationState):
        pass # pragma: no cover

    def pre_run(
        self, initial_state: SimulationState, structure: PeriodicStructure = None
    ) -> None:
        pass

    def get_random_site(self):
        return random.randint(0, len(self.structure.site_ids) - 1)

    def instantiate_result(self, starting_state):
        return SimulationResult(starting_state=starting_state)
