import typing
from ..core import BasicStepArtist
from ..core import BasicSimulationStep
from ..core import COLORS

from .phase_map import PhaseMap

def get_color_map_from_step(simulation_step: BasicSimulationStep):
    color_map = {}

    for row in range(0, simulation_step.size):
        for col in range(0, simulation_step.size):

            cell_state = simulation_step.state[row, col]
            if cell_state not in color_map:
                color_map[cell_state] = COLORS[color_ct]
                color_ct += 1

class DiscreteStepArtist(BasicStepArtist):

    def __init__(self, phase_map: PhaseMap, color_map = None):
        if color_map is None:
            self.color_map = self._phase_color_map(phase_map.phases)
        else:
            self.color_map = color_map
        self.phase_map = phase_map

    def get_color_by_cell_state(self, cell_state):
        phase_name = self.phase_map.int_to_phase[cell_state]
        return self.color_map[phase_name]

    def _phase_color_map(self, phase_list) -> typing.Dict[str, typing.Tuple[int, int ,int]]:
        """Returns a map of phases to colors that can be used to visualize the phases

        Returns:
            typing.Dict[str, typing.Tuple[int, int ,int]]: A mapping of phase name to RGB
            color values
        """
        display_phases: typing.Dict[str, typing.Tuple[int, int, int]] = {}
        c_idx: int = 0
        for p in phase_list:
            display_phases[p] = COLORS[c_idx]
            c_idx += 1

        return display_phases

    def get_legend(self):
        return self.color_map

