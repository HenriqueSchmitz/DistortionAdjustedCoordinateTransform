from setuptools import setup, find_packages

setup(
    name='DistortionAdjustedCoordinateTransform',
    version='1.0.0',
    author="Henrique Schmitz",
    packages=find_packages(),
    py_modules=['DistortionAdjustedCoordinateTransform'],
    install_requires=["typing", "torch"],
    setup_requires=["typing", "torch"],
)