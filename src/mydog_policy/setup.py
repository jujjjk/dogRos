from glob import glob

from setuptools import find_packages, setup

package_name = 'mydog_policy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mydog_policy_node = mydog_policy.mydog_policy_node:main',
            'mydog_state_estimator_node = mydog_policy.state_estimator_node:main',
            'mydog_openloop_gait_node = mydog_policy.openloop_gait_node:main',
            'fanfan_ik_gait_node = mydog_policy.fanfan_ik_gait_node:main',
            'fanfan_step_in_place_node = mydog_policy.fanfan_step_in_place_node:main',
            'fanfan_cpg_vmc_v4_node = mydog_policy.fanfan_cpg_vmc_v4_node:main',
            'fanfan_cpg_vmc_v4_migration_node = mydog_policy.fanfan_cpg_vmc_v4_migration_node:main',
            'mydog_soft_body_gait_node = mydog_policy.openloop_gait_node_soft_body_urdf:main',
            'mydog_joint_semantic_test_node = mydog_policy.joint_semantic_test_node:main',
        ],
    },
)
