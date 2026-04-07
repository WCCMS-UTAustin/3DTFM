from setuptools import find_packages, setup

setup(
    name="gel_model",
    version="0.0.4",
    packages=find_packages(),
    entry_points={
        "console_scripts" : [
            "forward_hydrogel = gel.scripts:forward",
            "inverse_hydrogel = gel.scripts:inverse",
            "downsample_mesh_hydrogel = gel.scripts:downsample_mesh_main",
            "get_u_hydrogel = gel.scripts:get_u_main",
            "get_kinematics_mesh_hydrogel = gel.scripts:get_kinematics_mesh",
            "get_veh_hydrogel = gel.scripts:get_veh_main",
            "get_displacement_from_gpr_hydrogel = gel.scripts:get_u_from_gpr_main",
            "create_cell_surf_normals_mesh_hydrogel = gel.scripts:create_cell_surf_normals_mesh_main",
        ]
    }
)

