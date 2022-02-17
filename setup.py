from setuptools import setup, find_packages

setup(
    name="ParallelLinear",
    version="0.0.1",
    author="Craig Ramage",
    description="Linear Maths Library",
    packages=['parallellinear.calculations', 'parallellinear.datatypes'],
    include_package_data=True,
    install_requires=[
        "numpy==1.22.2",
        "platformdirs==2.5.0",
        "pyopencl==2022.1",
        "pytools==2022.1",
    ]
)