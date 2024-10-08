from setuptools import setup, find_namespace_packages
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR,  'requirements.txt')).read().splitlines()

setup(
    name="PhaseMotif",
    version="0.1",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=INSTALL_PACKAGES,
    python_requires='>=3.8',
)
