import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lab03_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yousef, giorgia, Yijing, Adnan',
    maintainer_email='s355447@studenti.polito.it, s352186@studenti.polito.it, s353515@studenti.polito.it, s355177@studenti.polito.it',
    description='This package implements a bump-and-go controller for TurtleBot3 in a real life scenarios.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bump_go_real = lab03_pkg.bump_go_real:main',
        ],
    },
)
