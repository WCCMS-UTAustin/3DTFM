#!/usr/bin/env python3
from dolfin import *
import argparse
import os
import subprocess
import io
import numpy as np


def get_exp_u_xdmf(
        cell_data_dir,
        gpr_dir,
        outfile
    ):
    # Gel Volume Mesh
    mesh = Mesh()
    with XDMFFile(
        os.path.join(cell_data_dir, "reference_domain.xdmf")
    ) as infile:
        infile.read(mesh)

    # Get vertices
    gel_vertices = mesh.coordinates()

    #
    # Pass vertices to GPR subprocess
    #
    # Write to IO object
    s = io.BytesIO()
    np.savetxt(s, gel_vertices)

    # Spawn process to use GPR model
    script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "get_displacement_from_gpr.py"
    )
    conda_env = os.path.join(
        os.path.dirname(os.environ["CONDA_PREFIX"]),
        "scikit"
    )
    script_cmdline = (
        f"conda run --no-capture-output -p {conda_env} {script_path}"
        f" -v /dev/stdin -d /dev/stdout -g "+'"'+f"{gpr_dir}"+'"'
    )
    completed_proc = subprocess.run(
        script_cmdline,
        shell=True,
        input=s.getvalue(),
        capture_output=True
    )

    # Read in from IO object
    gel_displacements = np.loadtxt(io.BytesIO(completed_proc.stdout))
    print(completed_proc.stderr)

    # Convert to FE function, then output
    V = VectorFunctionSpace(mesh, "P", 1, dim=3)
    v2d = vertex_to_dof_map(V)
    u = Function(V)

    gel_displacements_flat = gel_displacements.flatten()
    u.vector()[v2d] = gel_displacements_flat

    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    u = project(u, V, solver_type='cg', preconditioner_type='amg')
    u.rename("u", "displacement")

    if outfile is None:
        outfile = os.path.join(cell_data_dir, "u_experiment.xdmf")
    u_outfile = XDMFFile(outfile)
    u_outfile.write_checkpoint(u, "u", 0) # Not appending


def get_u_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        type=str,
        metavar="CELL_DATA_DIR"
    )
    parser.add_argument(
        "-g",
        type=str,
        metavar="GPR_DIR"
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="OUT_FILE",
        default=None
    )
    args = parser.parse_args()

    get_exp_u_xdmf(args.c, args.g, args.o)


if __name__=="__main__":
    get_u_main()

