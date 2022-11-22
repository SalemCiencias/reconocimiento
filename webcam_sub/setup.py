from setuptools import setup

package_name = 'webcam_sub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cocisran',
    maintainer_email='fcoemmdmm@ciencias.unam.mx',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cam_pub = webcam_sub.webcam_pub:main',
            'cam_sub = webcam_sub.webcam_sub:main',
        ],
    },
)
