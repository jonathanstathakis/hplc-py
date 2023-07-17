from setuptools import setup, find_packages
import pathlib

__version__ = "0.0.01"

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="hplc-py",
    version=__version__,
    long_description=README,
    description="Python utilities for the processing and quantification of chromatograms from High Performance Liquid Chromatography (HPLC).",
    long_description_content_type='text/markdown',
    url="https://github.com/cremerlab/hplc-py",
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    author="Griffin Chure",
    author_email="griffinchure@gmail.com",
    packages=find_packages(exclude=('docs', 'docsrc', 'exploratory', 'cremerlab.egg-info')),
    include_package_data=True
)
