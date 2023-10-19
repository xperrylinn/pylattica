from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import rustworkx as rx
from tqdm import tqdm

from .constants import LOCATION, SITE_CLASS, SITE_ID
from .distance_map import EuclideanDistanceMap
from .neighborhoods import Neighborhood
from .periodic_structure import PeriodicStructure
from .lattice import pbc_diff_cart


class NeighborhoodBuilder(ABC):
    @abstractmethod
    def get(self, struct: PeriodicStructure) -> Neighborhood:
        pass # pragma: no cover


class DistanceNeighborhoodBuilder(NeighborhoodBuilder):
    """This neighborhood builder creates neighbor connections between
    sites which are within some cutoff distance of eachother.
    """

    def __init__(self, cutoff: float):
        """Instantiates a DistanceNeighborhoodBuilder

        Parameters
        ----------
        cutoff : float
            The maximum distance at which two sites are considered neighbors.
        """
        self.cutoff = cutoff

    def get(self, struct: PeriodicStructure) -> Neighborhood:
        """Builds a NeighborGraph from the provided structure according
        to the cutoff distance of this Builder.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure from which a NeighborGraph should be constructed.

        Returns
        -------
        NeighborGraph
            The resulting NeighborGraph
        """
        graph = rx.PyGraph()

        all_sites = struct.sites()

        for site in struct.sites():
            graph.add_node(site[SITE_ID])

        for curr_site in tqdm(all_sites):
            for other_site in struct.sites():
                if curr_site[SITE_ID] != other_site[SITE_ID]:
                    dist = pbc_diff_cart(
                        np.array(other_site[LOCATION]),
                        np.array(curr_site[LOCATION]),
                        struct.lattice,
                    )
                    if dist < self.cutoff:
                        graph.add_edge(curr_site[SITE_ID], other_site[SITE_ID], dist)

        return Neighborhood(graph)


class StructureNeighborhoodBuilder(NeighborhoodBuilder):
    """This NeighborhoodBuilder constructs NeighborGraphs with connections between
    points that are separated by one of a set of specific offset vectors.

    For example, consider a 2D structure with two types of sites, A and B. Each A
    site is connected to two other A sites, one offset by 1 unit in each of the positive
    and negative x directions. Each A site is also connected to two B sites, one offset
    by one unit in each of the positive and negative y directions, then the spec
    parameter for this arrangement would look as follows.

    {
        "A": [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ],
        "B": [
            [0, 1],
            [0, -1],
        ]
    }

    Note that there is reciprocity here between the A and B sites. The A sites
    list B sites as their neighbors, and the B sites list A sites as their neighbors.
    """

    def __init__(self, spec: Dict[str, List[List[float]]]):
        """Instantiates the StructureNeighborhoodBuilder by a spec as described in
        the docstring for the class.

        Parameters
        ----------
        spec : Dict[str, List[List[float]]]
            See class docstring.
        """
        self._spec = spec

        neighbor_locs = [loc for loclist in spec.values() for loc in loclist]
        self.distances = EuclideanDistanceMap(neighbor_locs)

    def get(self, struct: PeriodicStructure) -> Neighborhood:
        """Given a structure, constructs a NeighborGraph with site connections
        according to the spec.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which a NeighborGraph should be constructed.

        Returns
        -------
        NeighborGraph
            The resulting NeighborGraph.
        """
        graph = rx.PyGraph()

        for site in struct.sites():
            graph.add_node(site[SITE_ID])

        all_sites = struct.sites()
        for site in tqdm(all_sites):
            site_class = site[SITE_CLASS]
            location = site[LOCATION]
            site_class_neighbors = self._spec[site_class]
            edges = []
            for neighbor_vec in site_class_neighbors:
                loc = tuple(s + n for s, n in zip(location, neighbor_vec))
                nb_site = struct.site_at(loc)
                if nb_site[SITE_ID] != site[SITE_ID]:
                    edges.append(
                        (
                            nb_site[SITE_ID],
                            site[SITE_ID],
                            self.distances.get_dist(neighbor_vec),
                        )
                    )
            graph.add_edges_from(edges)

        return Neighborhood(graph)
