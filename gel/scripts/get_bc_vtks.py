#!/usr/bin/env python3
"""Generates bci.vtk, bco.vtk with FM-Track GPR-interpolation.

Must be run in the FM-Track environment. Will use a GPR model saved to
a directory with a specific format readable by the functions in
`get_displacements_from_gpr.py`.
"""
import argparse
import os
import numpy as np
import meshio
from .get_displacement_from_gpr import get_predicted_u


def my_optimizer(obj_func, initial_theta, bounds):
    import scipy
    opt_res = scipy.optimize.minimize(
        obj_func, initial_theta, method="L-BFGS-B", jac=True,
        bounds=bounds, options={"maxiter" : 3, "disp" : True})
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min


def _write_bc_vtk(gpr_dir, gel_mesh_boundaries, boundary_ind, filepath):
    boundary_mask = np.zeros(len(gel_mesh_boundaries.points)).astype(bool)
    boundary_mask[
        gel_mesh_boundaries.cells[0].data[
            gel_mesh_boundaries.cell_data["boundaries"][0] == boundary_ind
        ]
    ] = True

    boundary_points = gel_mesh_boundaries.points[boundary_mask]

    u = get_predicted_u(gpr_dir, boundary_points)

    mesh = meshio.Mesh(boundary_points, {}, point_data={"u":u})
    mesh.write(filepath)


def get_bc_vtks(
        cell_data_dir,
        gpr_dir,
        out_dir
    ):
    """Writes files `bci.vtk` and `bco.vtk` under `out_dir` for `gel.Geometry`

    * `cell_data_dir`: str path to directory with
    `reference_domain_boundaries.xdmf`
    * `gpr_dir`: str path to directory with GPR model files
    * `out_dir`: str path to directory in which to place `bci.vtk` and 
    `bco.vtk`

    Side-effects: writes files as specified
    """
    gel_mesh_boundaries = meshio.read(
        os.path.join(cell_data_dir, "reference_domain_boundaries.xdmf")
    )

    _write_bc_vtk(
        gpr_dir,
        gel_mesh_boundaries,
        201,
        os.path.join(out_dir, "bco.vtk")
    )
    _write_bc_vtk(
        gpr_dir,
        gel_mesh_boundaries,
        202,
        os.path.join(out_dir, "bci.vtk")
    )


def get_bc_vtks_main():
    """The function invoked by the command. Parses arguments and passes
    to `get_bc_vtks`.
    """
    parser = argparse.ArgumentParser(
        description="Generates boundary condition files bci.vtk and "
        "bco.vtk with FM-Track GPR-interpolation. Must be run in "
        "FM-Track environment. Places files with those names in "
        "specified directory. Will determine boundaries by "
        "CELL_DATA_DIR/reference_domain_boundaries.xdmf"
    )
    parser.add_argument(
        "-c",
        type=str,
        metavar="CELL_DATA_DIR",
        help="directory containing gel geometry, namely "
        "reference_domain_boundaries.xdmf and .h5"
    )
    parser.add_argument(
        "-g",
        type=str,
        metavar="GPR_DIR",
        help="directory with GPR model files, i.e. gp_U_cleaned.sav etc."
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="OUT_DIR",
        default=None,
        help="output directory in which to put bci.vtk and bco.vtk, "
        "default CELL_DATA_DIR"
    )
    args = parser.parse_args()

    if args.o is None:
        args.o = args.c

    get_bc_vtks(args.c, args.g, args.o)


if __name__=="__main__":
    get_bc_vtks_main()

