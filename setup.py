from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'stretch_dual_lidar_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'opencv-python', 'small_gicp'],
    zip_safe=True,
    maintainer='Hello Robot Inc.',
    maintainer_email='support@hello-robot.com',
    description='Calibration packages for Stretch dual lidar configuration',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_align_dual_lidar = stretch_dual_lidar_calibration.ros_align_dual_lidar:main',
            'ros_find_floor_calibration = stretch_dual_lidar_calibration.ros_find_floor_calibration:main',
            'ros_find_body_calibration = stretch_dual_lidar_calibration.ros_find_body_calibration:main',
            'ros_broadcast_calibration = stretch_dual_lidar_calibration.ros_broadcast_calibration:main',
            'ros_visualize_calibration = stretch_dual_lidar_calibration.ros_visualize_calibration:main',
            'ros_collect_body_shape_data = stretch_dual_lidar_calibration.ros_collect_body_shape_data:main',
            'ros_visualize_body_shape_calibration = stretch_dual_lidar_calibration.ros_visualize_body_shape_calibration:main',
            'fit_body_shape_model = stretch_dual_lidar_calibration.fit_body_shape_model:main',
        ],
    },
)
