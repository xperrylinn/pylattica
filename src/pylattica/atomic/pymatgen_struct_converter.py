from pymatgen.core.structure import Structure as PmgStructure
from pymatgen.core.lattice import Lattice as PmgLattice

from ..core import Lattice as PylLattice, StructureBuilder, PeriodicStructure


class PymatgenStructureConverter():
    def to_pylattica_lattice(self, pmg_lat: PmgLattice):
        pmg_lat = pmg_lat
        pyl_lat = PylLattice(pmg_lat.matrix)
        return pyl_lat

    def to_pylattica_structure_builder(self, pmg_struct: PmgStructure):
        lat = self.to_pylattica_lattice(pmg_struct.lattice)

        struct_motif = {}

        for site in pmg_struct.sites:
            site_cls = site.species_string
            frac_coords = site.frac_coords

            if site_cls in struct_motif:
                struct_motif.get(site_cls).append(frac_coords)
            else:
                struct_motif[site_cls] = [frac_coords]

        struct_builder = StructureBuilder(lat, struct_motif)
        struct_builder.frac_coords = True
        return struct_builder

    def to_pymatgen_lattice(self, pyl_lat: PylLattice):
        return PmgLattice(pyl_lat.vecs)

    def to_pymatgen_structure(self, pyl_struct: PeriodicStructure):
        pmg_lat = self.to_pymatgen_lattice(pyl_struct.lattice)

        species = []
        coords = []

        for sid in pyl_struct.site_ids:
            species.append(pyl_struct.site_class(sid))
            coords.append(
                pyl_struct.lattice.get_fractional_coords(pyl_struct.site_location(sid))
            )

        return PmgStructure(pmg_lat, species, coords)
