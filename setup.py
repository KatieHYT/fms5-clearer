import setuptools
__version__ = "0.0.6"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fms5clearer",
    version=__version__,
    author="kthuangyuting",
    author_email="yuting2727@gmail.com",
    description="An updated solution for satellite images of Formosa5.",
    long_description=long_description,
    url="https://github.com/katiehyt/fms5-clearer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
