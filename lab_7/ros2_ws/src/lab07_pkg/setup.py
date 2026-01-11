from setuptools import find_packages, setup

package_name = 'lab07_pkg'

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
    maintainer='yousef, giorgia, Yijing, Adnan',
    maintainer_email='s355447@studenti.polito.it, s352186@studenti.polito.it, s353515@studenti.polito.it, s355177@studenti.polito.it',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dwa_node = lab07_pkg.dwa_node:main',
            'adwa_node = lab07_pkg.adwa_node:main',
            'adwa_robot_node = lab07_pkg.adwa_robot_node:main',
            'dwa_analysis_node = lab07_pkg.dwa_analysis_node:main',
        ],
    },
)
