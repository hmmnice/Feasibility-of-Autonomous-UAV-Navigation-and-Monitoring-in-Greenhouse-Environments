import glob

from setuptools import find_packages, setup


package_name = "greenhouse_inspection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch",
         glob.glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yash Bhatia",
    maintainer_email="yash.bhatia.25@ucl.ac.uk",
    description="Map-informed UAV crop-inspection pipeline for a Venlo greenhouse.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "generate_static_map = "
            "greenhouse_inspection.static_greenhouse:main",
            "select_crop = greenhouse_inspection.target_selector:main",
            "gps_preflight = greenhouse_inspection.gps_preflight:main",
            "gps_auto_preflight = "
            "greenhouse_inspection.gps_auto_preflight:main",
            "identity_alignment = greenhouse_inspection.identity_alignment:main",
            "execute_viewpoint = greenhouse_inspection.execute_viewpoint:main",
            "execute_two_viewpoints = "
            "greenhouse_inspection.two_viewpoint_mission:main",
            "build_mission_plan = "
            "greenhouse_inspection.mission_plan:main",
            "gimbal_controller = "
            "greenhouse_inspection.gimbal_controller:main",
        ],
    },
)
