"""Interface to hydrogel geometry and BC information, reading required files"""
from .header import *

from numbers import Number as NumberType
from itertools import product
from scipy.spatial import distance_matrix

from .helper import *


def _give_me_disp(displacement, V, nodes):
    # Init
    u_meas = Function(V)

    coords = V.tabulate_dof_coordinates()

    under_const = np.zeros(len(coords))
    for i in range(len(coords)):
        this_coord = coords[i]
        dist_mat = distance_matrix(this_coord.reshape((1,3)), nodes)
        if np.min(dist_mat) < 1e-6:
            disp = displacement[np.argmin(dist_mat)]
            under_const[i] = disp[i % 3]
    u_meas.vector().set_local(under_const)
    u_meas.vector().apply("")

    return u_meas


class _DisplacementBasedSubdomain(SubDomain):

    _tol = 1e-14

    def __init__(self, u, lower_bound, upper_bound):
        self.u = u
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def inside(self, x, on_boundary):
        u_here = np.linalg.norm(self.u(x))
        return (
            (self.lower_bound - self._tol < u_here)
            and (u_here < self.upper_bound + self._tol)
        )


class Geometry:

    CUBE = 201
    """Subdomain marker for far edges of gel box"""
    CELL = 202
    """Subdomain marker for inner cell surface"""
    UNDETECTABLE_U = 250
    """Subdomain marker for volume outside event horizon, when applicable"""
    DETECTABLE_U = 251
    """Subdomain marker for volume within event horizon, when applicable"""

    def __init__(
            self,
            data_directory,
            xdmf_name="reference_domain",
            suppress_cell_bc=False,
            u_magnitude_subdomains_file=None,
            detectable_u_cutoff=0.38, # microns
            bci_data=None,
            bco_data=None
        ):
        """
        Object that contains data pertaining to geometry and BCs.

        * `data_directory`: str path to a directory containing required
        files listed below.
        * `xdmf_name`: str name of an .xdmf file in `data_directory`
        containing FEniCS-compatible geometry with subdomain markers
        and the variant with boundary subdomain markers.
        * `suppress_cell_bc`: bool, enables removing BC on the cell
        surface for Neumann BC
        * `u_magnitude_subdomains_file`: str path to full-shape .xdmf
        file with displacements labeled "u" from which the event horizon
        is determined (None => no event horizon used)
        * `detectable_u_cutoff`: float cutoff value for event horizon
        if such functionality enabled by supplying a
        `u_magnitude_subdomains_file`
        * `bci_data`: str or None. Describes the inner cell surface
        boundary condition. If None, looks inside `data_directory` for
        "bci.vtk" with point_data "u" on cell surface. Otherwise, looks
        for same under path to a .vtk file.
        * `bco_data`: str or None. Describes the outer box boundary
        condition. If None, sets the BC to 0 displacement. If a str,
        interprets as a path to a .vtk file with displacements in "u"
        point_data.

        Reads relevant data from files in `data_directory`.
        Required files:
        * {xdmf_name}.xdmf
        * {xdmf_name}.h5
        * {xdmf_name}_boundaries.xdmf
        * {xdmf_name}_boundaries.h5
        * `bci_data` .vtk file
        * If `bco_data` is not None, `bco_data` .vtk file
        """
        # Helper to gel relative path
        rp = ghrp(data_directory)

        # Gel Volume Mesh
        mesh = Mesh()
        with XDMFFile(rp(f"{xdmf_name}.xdmf")) as infile:
            infile.read(mesh)

        # Boundary info
        mvc = MeshValueCollection("size_t", mesh, 2)
        with XDMFFile(rp(f"{xdmf_name}_boundaries.xdmf")) as infile:
            infile.read(mvc, "boundaries") 
        boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

        # Function Spaces
        V0 = FunctionSpace(mesh, "Lagrange", 1)
        V = VectorFunctionSpace(mesh, "Lagrange", 1)
        VC = TensorFunctionSpace(mesh, "CG", degree=1, shape=(3,3))
        self._DG0 = None

        # Create subdomains if requested, create volumetric measure
        if u_magnitude_subdomains_file is not None:
            # Load in u
            u = Function(V)  

            u_file = XDMFFile(u_magnitude_subdomains_file)
            u_file.read_checkpoint(u, "u", 0)

            u.set_allow_extrapolation(True)
            u = interpolate(u, V)
            self.u_target = u

            # Create subdomains
            regions = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            
            # Chose 40 microns since near size of cell
            self.detectable_u_cutoff = detectable_u_cutoff
            detectable_subdomain = _DisplacementBasedSubdomain(
                u,
                detectable_u_cutoff,
                40
            )

            # Manually assign values to region due to need for alternative
            vertex_coords = mesh.coordinates()
            for ci, cell in enumerate(mesh.cells()):
                point_array = vertex_coords[cell]

                in_detectable_u = False
                for point_coord in point_array:
                    x = Point((*point_coord,))
                    if detectable_subdomain.inside(x, False):
                        # Only need a single point inside to count
                        # Note default .mark behavior wants all vertices inside
                        # and some other unknown criteria (or has a bug)
                        in_detectable_u = True
                        break

                regions.array()[ci] = (self.DETECTABLE_U if in_detectable_u 
                                       else self.UNDETECTABLE_U)

            dx = Measure("dx", domain=mesh, subdomain_data=regions)
        else:
            dx = Measure("dx", domain=mesh)

        # Surface measure
        ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

        #
        # Boundary Conditions
        #
        # Outer
        zero = Constant((0.0, 0.0, 0.0))

        if bco_data is None:
            # Default case: 0
            outer_bc = DirichletBC(V, zero, boundaries, 201)
        else:
            outer_mesh = meshio.read(bco_data)
            outer_surface_nodes = outer_mesh.points
            displacements = outer_mesh.point_data["u"]
            outer_bf = _give_me_disp(displacements, V, outer_surface_nodes)

            outer_bc = DirichletBC(V, outer_bf, boundaries, 201)

        bcs = [outer_bc]

        # Inner
        if not suppress_cell_bc:
            if bci_data is None:
                # Default case: look for bci.vtk
                bci_data = rp("bci.vtk")

            surf_mesh = meshio.read(bci_data)
            surface_nodes1 = surf_mesh.points
            displacements = surf_mesh.point_data["u"]

            self.cell_vertices = surface_nodes1

            self.bf = _give_me_disp(
                displacements,
                V,
                surface_nodes1
            )
            self.scalar = 0.0

            inner_bc = DirichletBC(V, zero, boundaries, 202)
            bcs.append(inner_bc)

        # Set to internal variables
        self.V0 = V0
        """Scalar 1st order Lagrange function space on gel mesh"""
        self.V = V
        """Vector 1st order Lagrange function space on gel mesh"""
        self.VC = VC
        """2nd rank tensor 1st order Lagrange function space on gel mesh"""

        self.dx = dx
        """Volumetric measure on gel mesh with subdomain data"""
        self.ds = ds
        """Surface measure on gel mesh boundaries with subdomain data"""

        self.bcs = bcs
        """List of FEniCS boundary conditions. 0 is outer, 1 is inner
        (if present)
        """

        self.mesh = mesh
        """Corresponding FEniCS mesh object."""

        self.boundaries = boundaries
        """Boundary MeshFunctionSizet with surface tags"""
        self._suppress_cell_bc = suppress_cell_bc

    def output_regions(self, filename):
        """Writes subdomain data to file under path filename."""
        regions = self.dx.subdomain_data()
        File(filename) << regions

    def update_bcs(self):
        """Updates `bf` according to `scalar` for load stepping.

        Side effects: inner cell surface boundary condition `bcs`
        updates according to float `scalar`
        """
        if not self._suppress_cell_bc:
            new_bc = Function(self.V)
            new_bc.vector().set_local(self.bf.vector().get_local()*self.scalar)
            new_bc.vector().apply("")
            self.bcs[1] = DirichletBC(self.V, new_bc, self.boundaries, 202)

    @property
    def DG0(self):
        """Element-wise basis/function space on the gel mesh"""
        if self._DG0 is None:
            self._DG0 = FunctionSpace(self.mesh, "DG", 0)
        return self._DG0

