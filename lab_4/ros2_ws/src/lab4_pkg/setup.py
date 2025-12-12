import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lab04_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yousef, giorgia, Yijing, Adnan',
    maintainer_email='s355447@studenti.polito.it, s352186@studenti.polito.it, s353515@studenti.polito.it, s355177@studenti.polito.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'task0_velocity = lab04_pkg.velocity_model:main',
            'task0_landmark = lab04_pkg.landmark_model:main',
            'task1_node = lab04_pkg.task1:main',
            'task2_node = lab04_pkg.task2_def:main',
        ],
    
    },
)
