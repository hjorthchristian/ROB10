from setuptools import setup, find_packages

package_name = 'image_processor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'ultralytics'],
    zip_safe=True,
    maintainer='chrishj',
    maintainer_email='cankhjorth@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_subscriber = image_processor.image_subscriber:main',
            'lang_sam_client = image_processor.lang_sam_client:main',
            'segmentation_and_pose_estimation = image_processor.segmentation_and_pose_estimation:main',
            'segmentation_and_pose_estimation_v2 = image_processor.segmentation_and_pose_estimation_v2:main',

        ],
    },
)
