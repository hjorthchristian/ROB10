from setuptools import find_packages, setup

package_name = 'pose_estimation'

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
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation = pose_estimation.pose_estimation_client:main',
            'pose_estimation_service = pose_estimation.pose_estimation_server:main',
        ],
    },
)
