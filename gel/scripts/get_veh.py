#!/usr/bin/env python3
import argparse
import os
os.environ["SUPPRESS_ADJOINT"] = "1"
from gel import *


def get_veh(cell_data_dir, u_exp_file, directory, cutoff=0.5):
    geo = Geometry(
        cell_data_dir,
        u_magnitude_subdomains_file=u_exp_file,
        detectable_u_cutoff=cutoff
    )
    if cutoff == 0.5:
        filename = os.path.join(directory, "u_regions.pvd")
    else:
        filename = os.path.join(directory, f"u_regions_{cutoff}.pvd")
    File(filename) << geo.dx.subdomain_data()


def get_veh_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        type=str,
        metavar="CELL_DATA"
    )
    parser.add_argument(
        "-i",
        type=str,
        metavar="INPUT_FULL_SHAPE"
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="OUTPUT_DIR"
    )
    parser.add_argument(
        "-t",
        type=float,
        metavar="THRESHOLD",
        default=0.5
    )
    args = parser.parse_args()

    get_veh(args.c, args.i, args.o, args.t)


if __name__=="__main__":
    get_veh_main()

