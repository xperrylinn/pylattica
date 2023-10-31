import pytest

from pylattica.atomic.pymatgen_struct_converter import PymatgenStructureConverter
from pylattica.core.constants import SITE_CLASS, LOCATION
from pymatgen.core.structure import Structure as PmgStructure
from pylattica.core import PeriodicStructure

import numpy as np

def test_can_convert_lattice(zr_pmg_struct: PmgStructure):
    converter = PymatgenStructureConverter()

    lattice = converter.to_pylattica_lattice(zr_pmg_struct.lattice)

    assert np.isclose(lattice.vec_lengths[0], zr_pmg_struct.lattice.a)
    assert np.isclose(lattice.vec_lengths[1], zr_pmg_struct.lattice.b)
    assert np.isclose(lattice.vec_lengths[2], zr_pmg_struct.lattice.c)

    assert np.allclose(lattice.vecs[0], zr_pmg_struct.lattice.matrix[0])
    assert np.allclose(lattice.vecs[1], zr_pmg_struct.lattice.matrix[1])
    assert np.allclose(lattice.vecs[2], zr_pmg_struct.lattice.matrix[2])

def test_can_convert_pmg_struct(zr_pmg_struct: PmgStructure):
    converter = PymatgenStructureConverter()

    struct_builder = converter.to_pylattica_structure_builder(zr_pmg_struct)

    struct = struct_builder.build(1)

    assert len(struct.site_ids) == zr_pmg_struct.num_sites

    species_label = zr_pmg_struct.sites[0].species_string
    assert struct.site_class(0) == species_label

    # brittle - relies on the order in which sites are enumerated
    assert np.allclose(struct.site_location(0),zr_pmg_struct.sites[0].coords)
    assert np.allclose(struct.site_location(1),zr_pmg_struct.sites[1].coords)

    assert np.allclose(struct.lattice.get_fractional_coords(struct.site_location(1)), zr_pmg_struct.sites[1].frac_coords)

def test_can_convert_pyl_lat(pyl_struct: PeriodicStructure):
    converter = PymatgenStructureConverter()

    pmg_lat = converter.to_pymatgen_lattice(pyl_struct.lattice)
    assert pmg_lat.a == pyl_struct.lattice.vec_lengths[0]
    assert pmg_lat.b == pyl_struct.lattice.vec_lengths[1]
    assert pmg_lat.c == pyl_struct.lattice.vec_lengths[2]

    assert np.allclose(pyl_struct.lattice.vecs[0], pmg_lat.matrix[0])
    assert np.allclose(pyl_struct.lattice.vecs[1], pmg_lat.matrix[1])
    assert np.allclose(pyl_struct.lattice.vecs[2], pmg_lat.matrix[2])

def test_can_convert_pyl_struct(pyl_struct: PeriodicStructure):
    converter = PymatgenStructureConverter()

    pmg_struct = converter.to_pymatgen_structure(pyl_struct)

    assert pmg_struct.num_sites == len(pyl_struct.site_ids)

    for site in pmg_struct.sites:
        matching_site = pyl_struct.site_at(site.coords)
        assert matching_site is not None
        assert matching_site[SITE_CLASS] == site.species_string

        assert np.all(pyl_struct.lattice.get_fractional_coords(matching_site[LOCATION]) == site.frac_coords)