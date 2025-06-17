#!/usr/bin/env python3
import argparse
import numpy as np
import os
import pickle


_cache = dict()
def get_predicted_u(gpr_path, vertices):
    # loads GPR models
    if gpr_path not in _cache:
        gp_U = pickle.load(open(os.path.join(gpr_path, 'gp_U_cleaned.sav'), 'rb'))
        gp_V = pickle.load(open(os.path.join(gpr_path, 'gp_V_cleaned.sav'), 'rb'))
        gp_W = pickle.load(open(os.path.join(gpr_path, 'gp_W_cleaned.sav'), 'rb'))
        scaler = pickle.load(open(os.path.join(gpr_path, 'scaler_cleaned.sav'),'rb'))
        _cache[gpr_path] = (gp_U, gp_V, gp_W, scaler)
    else:
        (gp_U, gp_V, gp_W, scaler) = _cache[gpr_path]

    input_arr = scaler.transform(vertices)

    N = 310000
    if len(input_arr) < N:
        u = gp_U.predict(input_arr)
        v = gp_V.predict(input_arr)
        w = gp_W.predict(input_arr)
        disp = np.column_stack((u,v,w))
    else:
        # Must stage it
        disp = np.zeros_like(input_arr)
        for batch_i in range(int(np.ceil(len(input_arr)/N))):
            u = gp_U.predict(input_arr[N*batch_i:N*(batch_i+1)])
            v = gp_V.predict(input_arr[N*batch_i:N*(batch_i+1)])
            w = gp_W.predict(input_arr[N*batch_i:N*(batch_i+1)])
            disp[N*batch_i:N*(batch_i+1)] = np.column_stack((u,v,w))

    return disp


def get_u_from_gpr_main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPR model at provided vertices, for use "
        "in FM-Track environment"
    )
    parser.add_argument(
        "-v",
        "--vertices-file",
        type=str,
        metavar="VERTICES_FILE"
    )
    parser.add_argument(
        "-d",
        "--dest",
        type=str,
        metavar="DEST"
    )
    parser.add_argument(
        "-g",
        "--gpr-dir",
        type=str,
        metavar="GPR_DIR"
    )
    args = parser.parse_args()

    vertices = np.loadtxt(args.vertices_file)
    u = get_predicted_u(args.gpr_dir, vertices)
    np.savetxt(args.dest, u)


if __name__=="__main__":
    get_u_from_gpr_main()

