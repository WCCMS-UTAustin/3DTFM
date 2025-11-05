#!/usr/bin/env python3
"""Compute nodal cell surface normals, save as .vtk file

Uses Tensorflow (must be installed) to compute nodal normals on cell
surface mesh, saves in provided directory.
"""
import sys
if "pdoc" not in sys.modules:
    from tensorflow_graphics.geometry.representation.mesh.normals import vertex_normals
import argparse
import os
import numpy as np
import meshio


def create_cell_surf_normals_mesh(cell_mesh_file, dest_dir):
    """Creates mesh .vtk file with nodal normals.

    * `cell_mesh_file`: str path to mesh file with downsampled,
    centered cell surface (for instance, an .stl)
    * `dest_dir`: str path to directory in which to create new mesh file

    Side-effects: writes `cell_surf_with_normal.vtk` under `dest_dir`
    """
    cell_surf_mesh = meshio.read(cell_mesh_file)
    
    cell_vertices = cell_surf_mesh.points
    cell_conn = cell_surf_mesh.cells[0].data
    
    # Outward normal
    cell_normals = -vertex_normals(cell_vertices, cell_conn) 
    
    cell_surf_mesh.point_data["n"] = cell_normals.numpy()
    
    cell_surf_mesh.write(os.path.join(dest_dir, "cell_surf_with_normal.vtk"))


def create_cell_surf_normals_mesh_main():
    """The function invoked by the command. Parses arguments and passes
    to `create_cell_surf_with_normals_mesh`.
    """
    parser = argparse.ArgumentParser(
        description="Uses Tensorflow (must be installed) to compute nodal normals on cell surface mesh, saves in provided directory."
    )
    parser.add_argument(
        "-c",
        type=str,
        metavar="CELL_MESH_FILE",
        help="input cell mesh file"
    )
    parser.add_argument(
        "-d",
        type=str,
        metavar="DEST_DIR",
        help="directory in which cell_surf_with_normal.vtk will be "
        "placed"
    )
    args = parser.parse_args()

    create_cell_surf_normals_mesh(args.c, args.d)


if __name__=="__main__":
    create_cell_surf_normals_mesh_main()

