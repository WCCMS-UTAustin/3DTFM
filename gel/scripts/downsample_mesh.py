#!/usr/bin/env python3
import argparse
import os
import pymeshlab


def downsample_mesh(num_faces,filename,outputname):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)

    # New version
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=num_faces,
        qualitythr=1,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True
    )

    # New version
    ms.apply_filter("meshing_re_orient_faces_coherently")

    ms.save_current_mesh(outputname)


def downsample_mesh_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=str,
        metavar="WORKING_DIR"
    )
    parser.add_argument(
        "-n",
        "--num-reduced-vertices",
        type=int,
        metavar="NUM_REDUCED_VERTICES",
        default=1100
    )
    args = parser.parse_args()

    downsample_mesh(args.num_reduced_vertices, os.path.join(args.d, "CytoD.stl"), os.path.join(args.d, "CytoD_downsampled.stl"))


if __name__=="__main__":
    downsample_mesh_main()

