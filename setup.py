import os
from setuptools import setup, find_packages


def read(fname):
    """Reads a file's contents as a string.
    Args:
        fname: Filename.
    Returns:
        File's contents.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


INSTALL_REQUIRES = [
    "numpy>=1.16.3",
    #"scipy>=1.2.1",
    #"six>=1.12.0",
    #"Theano>=1.0.4",
]


setup(name="Reverse Pendulum",
      author="Mahtokh Gh",
      packages=find_packages(),
      zip_safe=True,
      install_requires=INSTALL_REQUIRES,
      )
