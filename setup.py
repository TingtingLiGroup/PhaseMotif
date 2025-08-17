from setuptools import setup, find_namespace_packages
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR,  'requirements.txt'), encoding='utf-8').read().splitlines()

setup(
    name="PhaseMotif",
    version="1.0",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=INSTALL_PACKAGES,
    python_requires='>=3.8',
)
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121