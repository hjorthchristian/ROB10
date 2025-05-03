from setuptools import find_packages, setup

package_name = 'stack_optimization_ros'

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
    maintainer='chrishj',
    maintainer_email='cankhjorth@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stack_optimizer_client = stack_optimization_ros.stack_optimizer_client:main',
            'stack_optimizer_server = stack_optimization_ros.stack_optimizer_server:main',
            'stack_optimizer_visualization = stack_optimization_ros.stack_optimizer_visualization:main',
        ],
    },
)
