"""# Introduction

Welcome to `gel`, the library for inverse finite-element analysis of
compressible heterogeneous hydrogel nonlinear materials!

# Command Usage

The first way to use this code is through the command-line. Upon
installation, numerous scripts are added to PATH for easy usage
anywhere on your computer.

Usage information is available through `--help` arguments. Documentation
for each program is also available here in `gel.scripts`, including API
for alternative library usage.

Command names and corresponding source files:
* forward: `gel.scripts.forward`
* inverse: `gel.scripts.inverse`
* downsample_mesh: `gel.scripts.downsample_mesh`
* get_u: `gel.scripts.get_u_from_gel_mesh`
* get_kinematics_mesh: `gel.scripts.full_shape_to_nodal`
* get_veh: `gel.scripts.get_veh`
* create_cell_surf_normals_mesh: `gel.scripts.create_cell_surf_normals_mesh`

# Library Usage

The second way to use this code is as a Python library. This is
especially useful for post-processing in Jupyter notebooks, of which
you may find an
[example in the repository](https://github.com/WCCMS-UTAustin/3DTFM/blob/main/reproduction/fenics_environment_analysis.ipynb).

As with other FEniCS-based libraries, start with a star-import:

```
from gel import *
```

Then, you will be provided with all of the imports from FEniCS,
overloaded with FEniCS-adjoint under most circumstances, as well as,
principally, the contents of the following submodules:
* `gel.gel`
* `gel.geometry`
* `gel.kinematics`
* `gel.mechanics`
* `gel.objective`
* `gel.helper`

Typically, library usage is for exploration and post-processing. In this
case, geometry, kinematics, and mechanics are defined in that order.
In the case of post-procesing, this looks like:
```
geo = Geometry(cell_data_dir)
kin = kinematics_from_file(geo, u_file)
beta = load_shape(geo.V0, beta_file, "mod_repr")
mech = Mechanics(kin, mod_repr=beta)
```
"""
from .gel import *

