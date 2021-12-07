
from setuptools import setup, find_packages
import os, fnmatch, re, sys
from nclustenv.version import VERSION

license = ''
with open('nclustRL/__init__.py', 'r') as fd:
    content = fd.read()
    license = re.search(
        r'^__license__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

version = VERSION

if version is None:
    raise RuntimeError('Cannot find version information')

if license is None:
    raise RuntimeError('Cannot find license information')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nclustRL",
    version=version,
    author="Pedro Cotovio",
    author_email="pgcotovio@gmail.com",
    license=license,
    description="Toolbox to learn biclustering and triclustering task using Ray's rllib and torch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PedroCotovio/nclustRL",
    project_urls={
        "Bug Tracker": "https://github.com/PedroCotovio/nclustRL/issues",
        "Documentation": "https://nclustRL.readthedocs.org",
    },
    install_requires=[
        'ray>=1.9',
        'nclustenv>=0.1.0',
        'tqdm',
        'seaborn',
        'inquirer>=2.7',
        'numpy',
        'scipy'

    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',


        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',

        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    keywords='biclustring triclustering rl ray rllib torch data nclustRL',
    test_suite='tests',
)