from .header import *

from numbers import Number as NumberType
from itertools import product
from scipy.spatial import distance_matrix

from .helper import *


def give_me_disp(displacement, V, nodes):
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

    CUBE = 201 # Subdomain marker for far edges of gel cube
    CELL = 202 # Subdomain marker for inner cell surface
    UNDETECTABLE_U = 250 # Subdomain marker for undetectable disp (if requested)
    DETECTABLE_U = 251 # Subdomain marker for detectable disp (if requested)
    use_scaling = True

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
        Object that contains data pertaining to geometry being used.

        May optionally specify a xdmf_name different from default
        "reference_domain"

        Reads relevant data from files in data_directory.
        Needed files:
        * cell_vertices_initial.txt
        * cell_vertices_final.txt
        * cell_vertices_connectivity.txt
        * {xdmf_name}.xdmf
        * {xdmf_name}.h5
        * {xdmf_name}_boundaries.xdmf
        * {xdmf_name}_boundaries.h5
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
            outer_bc = DirichletBC(V, zero, boundaries, 201)
        else:
            outer_mesh = meshio.read(bco_data)
            outer_surface_nodes = outer_mesh.points
            displacements = outer_mesh.point_data["u"]
            outer_bf = give_me_disp(displacements, V, outer_surface_nodes)

            outer_bc = DirichletBC(V, outer_bf, boundaries, 201)

        bcs = [outer_bc]

        # Inner
        if not suppress_cell_bc:
            if bci_data is None:
                # Load in surface nodes ref/cur
                surface_nodes1 = np.loadtxt(rp("cell_vertices_initial.txt"))
                surface_nodes2 = np.loadtxt(rp("cell_vertices_final.txt"))

                # Surface Displacements
                displacements = surface_nodes2 - surface_nodes1
            else:
                surf_mesh = meshio.read(bci_data)
                surface_nodes1 = surf_mesh.points
                displacements = surf_mesh.point_data["u"]

            self.cell_vertices = surface_nodes1

            self.bf = give_me_disp(
                displacements,
                V,
                surface_nodes1
            )
            self.scalar = 0.0

            inner_bc = DirichletBC(V, zero, boundaries, 202)
            bcs.append(inner_bc)

        # Set to internal variables
        self.V0 = V0
        self.V = V
        self.VC = VC
        self.dx = dx
        self.ds = ds
        self.bcs = bcs
        self.mesh = mesh
        self.boundaries = boundaries
        self.suppress_cell_bc = suppress_cell_bc

    def output_regions(self, filename):
        regions = self.dx.subdomain_data()
        File(filename) << regions

    def update_bcs(self):
        if not self.suppress_cell_bc:
            new_bc = Function(self.V)
            new_bc.vector().set_local(self.bf.vector().get_local()*self.scalar)
            new_bc.vector().apply("")
            self.bcs[1] = DirichletBC(self.V, new_bc, self.boundaries, 202)

