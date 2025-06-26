"""Downsamples .stl file with `pymeshlab`"""
import argparse
import os
import pymeshlab


def downsample_mesh(num_faces,filename,outputname):
    """Reads fine-resolution file, writes coarse-resolution file.

    * `num_faces`: int number of target faces for downsampled mesh
    * `filename`: str path to fine-resolution .stl mesh
    * `outputname`: str path to coarse-resolution .stl mesh to create

    Side-effects: writes the new file
    """
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
    """The function invoked by the command. Parses arguments and passes
    to `downsample_mesh`.
    """
    parser = argparse.ArgumentParser(
        description="Interface to pymeshlab quadratic edge collapse for"
        " FEniCS environment usage"
    )
    parser.add_argument(
        "-d",
        type=str,
        metavar="WORKING_DIR",
        help="directory with input CytoD.stl and for output "
        "CytoD_downsampled.stl"
    )
    parser.add_argument(
        "-n",
        "--num-reduced-faces",
        type=int,
        metavar="NUM_REDUCED_FACES",
        default=1100,
        help="target number of triangular faces"
    )
    args = parser.parse_args()

    downsample_mesh(
        args.num_reduced_faces,
        os.path.join(args.d, "CytoD.stl"),
        os.path.join(args.d, "CytoD_downsampled.stl")
    )


if __name__=="__main__":
    downsample_mesh_main()

