r"""Interface to reading displacements from files, continuum mechanics.

The most useful aspect of this submodule is the class `Kinematics`
which contains definitions of various fields in continuum mechanics,
useful for defining material models in `gel.mechanics`. It also contains
the proper functionality for allowing differentiating through such
relationships to obtain stress tensors and for projecting $J$ into an
appropriate element-wise basis.
"""
from .header import *
from .helper import *
 

class Kinematics:

    def __init__(self, geo, u=None, differentiable_F=False):
        """
        Object that stores variables concerning the kinematics of the problem.

        Needs a geometry geo that all this movement is happening with, and may
        optionally provide a displacement u to impose. Otherwise, its u will
        be computed from scratch.

        In order to be able to differentiate computed quantities wrt. F, set
        differentiable_F=True , ie. for computing stress tensors.
        """
        self.geo = geo
        """`gel.geometry.Geometry` on which displacements are defined"""

        self.u = None
        r"""FEniCS function displacement field $\mathbf{u}$"""
        if u is None:
            self.u = Function(geo.V)
        else:
            self.u = u

        I = Identity(self.u.geometric_dimension())

        self.F = I + grad(self.u) 
        r"""$\mathbf{F}=\mathbf{I}+\nabla\mathbf{u}$"""

        self.differentiable_F = differentiable_F
        r"""bool indicating if `F` is equipped for being the argument
        that another form is differentiated with respect to
        """
        if differentiable_F:
            # Be able to use it as argument of differentiation
            self.F = variable(self.F)

        self.C = self.F.T*self.F # Right Cauchy-Green
        r"""$\mathbf{C}=\mathbf{F}^T\mathbf{F}$"""

        self.Ic = tr(self.C)
        r"""$I_1=\text{tr}\left(\mathbf{C}\right)$"""

        self.Ju = det(self.F)
        r"""$J=\det\left(\mathbf{F}\right)$"""

    @property
    def projected_J(self):
        r"""Element-wise-DoF $J$ FEniCS function"""
        return project(self.Ju, self.geo.DG0)


def kinematics_from_file(geo, filename, *args, **kwargs):
    """Returns `Kinematics` object from displacements "u" in xdmf `filename`.

    * `geo`: `gel.geometry.Geometry` object with underlying meshes
    matching that for the displacements in `filename`
    * `filename`: str path to full-shape/checkpoing .xdmf file with
    displacement data "U"
    * `args`, `kwargs`: other arguments for the `Kinematics` constructor

    Note that this will only read "u" at the first timestep.
    """
    u = load_shape(geo.V, filename, "u")

    return Kinematics(geo, *args, u=u, **kwargs)

