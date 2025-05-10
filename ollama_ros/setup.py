from setuptools import find_packages, setup

package_name = 'ollama_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='chrishj',
    maintainer_email='cankhjorth@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ollama_action_server = ollama_ros.ollama_action_server:main',
            'ollama_action_client = ollama_ros.ollama_action_client:main',
        ],
    },
)
