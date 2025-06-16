from .header import *
from sanity_utils import Stats
 

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

        if u is None:
            self.u = Function(geo.V)
        else:
            self.u = u

        d = self.u.geometric_dimension()
        self.I = Identity(d)

        # Deformation tensor
        self.F = self.I + grad(self.u) 
        self.differentiable_F = differentiable_F
        if differentiable_F:
            # Be able to use it as argument of differentiation
            self.F = variable(self.F)

        self.C = self.F.T*self.F # Right Cauchy-Green
        self.Ic = tr(self.C)
        self.Ju = det(self.F)


def kinematics_from_file(geo, filename, *args, **kwargs):
    """
    Given a path to an xdmf file which has full shape function outputs for u,
    displacements named u, and a matching geometry, creates a corresponding
    Kinematics object.

    May provide other arguments for the Kinematics constructor.

    Note that this will only read u at the first timestep.
    """
    u = Function(geo.V)  

    u_file = XDMFFile(filename)
    u_file.read_checkpoint(u, "u", 0)

    u.set_allow_extrapolation(True)
    u = interpolate(u, geo.V)

    return Kinematics(geo, *args, u=u, **kwargs)

