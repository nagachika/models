"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
        'avro-python3',
        'apache-beam',
        'pillow',
        'lxml',
        'matplotlib',
        'Cython',
        'contextlib2',
        'tf-slim',
        'six',
        'pycocotools',
        'scipy',
        'pandas',
        'tf-models-official==2.3.0'
        ]

setup(
    name='object_detection',
    version='0.6',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
)
