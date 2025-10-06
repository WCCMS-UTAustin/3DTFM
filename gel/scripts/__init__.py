"""All the functions associated with commands.

Command names and corresponding source files:
* forward: `gel.scripts.forward`
* inverse: `gel.scripts.inverse`
* downsample_mesh: `gel.scripts.downsample_mesh`
* get_u: `gel.scripts.get_u_from_gel_mesh`
* get_kinematics_mesh: `gel.scripts.full_shape_to_nodal`
* get_veh: `gel.scripts.get_veh`
"""
from .forward import forward
from .inverse import inverse
from .downsample_mesh import downsample_mesh_main
from .get_u_from_gel_mesh import get_u_main, get_exp_u_xdmf
from .full_shape_to_nodal import get_kinematics_mesh
from .get_veh import get_veh_main, get_veh
from .get_displacement_from_gpr import get_u_from_gpr_main

