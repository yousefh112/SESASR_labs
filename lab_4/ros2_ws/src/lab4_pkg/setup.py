from setuptools import find_packages, setup

package_name = 'lab4_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yousef',
    maintainer_email='yousef_hesham.yh112@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'task0_motion = lab4_pkg.task0_motion_sampler:main',
            'task0_landmark = lab4_pkg.task0_landmark_sampler:main',
            'compute_jacobians = lab4_pkg.compute_jacobians:main',
            'ekf_node = lab4_pkg.ekf_node:main',
            'ekf_node_task2 = lab4_pkg.ekf_node_task2:main',
        ],
    },
)
