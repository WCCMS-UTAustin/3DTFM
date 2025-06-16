from setuptools import find_packages, setup

setup(
    name="gel_model",
    version="0.0.4",
    packages=find_packages(),
    entry_points={
        "console_scripts" : [
            "forward = gel.scripts:forward",
            "inverse = gel.scripts:inverse",
            "downsample_mesh = gel.scripts:downsample_mesh_main",
            "get_u = gel.scripts:get_u_main",
            "get_kinematics_mesh = gel.scripts:get_kinematics_mesh",
            "get_veh = gel.scripts:get_veh_main",
            "get_displacement_from_gpr = gel.scripts:get_u_from_gpr_main"
        ]
    }
)

