r"""Implementation of `get_kinematics_mesh` command for $J$ postpro.
"""
import argparse
import os
os.environ["SUPPRESS_ADJOINT"] = "1"
from gel import *


def main(cell_data, input_full_shape, output_nodal):
    r"""Writes .xdmf with element-wise kinematics information.

    * `cell_data`: str path to directory with files needed by
    `gel.geometry.Geometry`
    * `input_full_shape`: str path to full-shape .xdmf file with
    displacements "u"
    * `output_nodal`: str path to .xdmf file (ending in that extension)
    that will be created

    Side-effects: writes `output_nodal` .xdmf and .h5 versions with
    kinematic information:
    * "u": $\mathbf{u}$
    * "J": $J$
    * "CC": $\mathbf{C}:\mathbf{C}$
    * "w2": $\mathbf{C}:\mathbf{C}-2\text{tr}\mathbf{C}+3$
    * "w3": $3\mathbf{C}:\mathbf{C}-\left(\text{tr}\mathbf{C}\right)^2$
    """
    geo = Geometry(cell_data)
    kinematics = kinematics_from_file(geo, input_full_shape)

    simulation_output_file = XDMFFile(output_nodal)
    simulation_output_file.parameters["flush_output"] = True
    simulation_output_file.parameters["functions_share_mesh"] = True

    kinematics.u.rename("u","displacement")

    DG_0 = FunctionSpace(geo.mesh, "DG", 0)
    J = project(kinematics.Ju, DG_0)
    J.rename("J","Jacobian")
    
    C_norm_sq = inner(kinematics.C, kinematics.C)
    C_ns_vec = project(C_norm_sq, DG_0)
    C_ns_vec.rename("CC","CNormSquared")

    w22 = C_norm_sq - (2*kinematics.Ic) + 3
    w2 = sqrt(conditional(gt(w22,0),w22,0))
    w2_vec = project(w2, DG_0)
    w2_vec.rename("w2","Weight2")

    w33 = (3*C_norm_sq) - (kinematics.Ic*kinematics.Ic)
    w3 = sqrt(conditional(gt(w33,0),w33,0))
    w3_vec = project(w3, DG_0)
    w3_vec.rename("w3","Weight3")

    # Writes out the variables
    simulation_output_file.write(kinematics.u,0)
    simulation_output_file.write(J,0)
    simulation_output_file.write(C_ns_vec,0)
    simulation_output_file.write(w2_vec,0)
    simulation_output_file.write(w3_vec,0)


def get_kinematics_mesh():
    """The function invoked by the command. Parses arguments and passes
    to `main`.
    """
    parser = argparse.ArgumentParser(
        description="Convert a full-shape .xdmf file with displacements"
        " to a nodal .xdmf file for both visualization and computing J"
    )
    parser.add_argument(
        "-c",
        type=str,
        metavar="CELL_DATA",
        help="directory containing gel geometry"
    )
    parser.add_argument(
        "-i",
        type=str,
        metavar="INPUT_FULL_SHAPE",
        help="full-shape 1st order Lagrange .xdmf with displacements 'u'"
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="OUTPUT_NODAL",
        help="output .xdmf file with kinematic quantities like J"
    )
    args = parser.parse_args()

    main(args.c, args.i, args.o)


if __name__=="__main__":
    get_kinematics_mesh()

