#!/usr/bin/env python3
"""Within the FM-TRACK environment, perform GPR-interpolation with text
input and output.

For all arguments, run `get_displacement_from_gpr --help`
"""
import argparse
import numpy as np
import os
import pickle


def my_optimizer(obj_func, initial_theta, bounds):
    """Function sometimes used for kernel hyperparameter optimization"""
    import scipy
    opt_res = scipy.optimize.minimize(
        obj_func, initial_theta, method="L-BFGS-B", jac=True,
        bounds=bounds, options={"maxiter" : 3, "disp" : True})
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min


_cache = dict()
def get_predicted_u(gpr_path, vertices):
    """Evaluates GPR model from directory.

    * `gpr_path`: str path to directory with files:
        * gp_U_cleaned.sav: FM-Track GPR model of x1-axis displacements
        * gp_V_cleaned.sav: FM-Track GPR model of x2-axis displacements
        * gp_W_cleaned.sav: FM-Track GPR model of x3-axis displacements
        * scaler_cleaned.sav: FM-Track GPR model pre-process component
    * `vertices`: (N,3) array of float coordinates in hydrogel

    Returns:
    * (N,3) array of float displacements at coordinates
    """
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
    """The function invoked by the command. Parses arguments and passes
    to `get_predicted_u`.
    """
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

