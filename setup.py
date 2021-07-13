import os
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name="nn_tuning",
    version="1.0.1",
    description="Analyse the tuning functions of neurons in artificial neural networks",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nn-analysis/nn_tuning",
    author="Jesca Hoogendijk",
    author_email="j.hoogendijk@uu.nl",
    license="GNU LGPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("tests", "prednet_fitting_example")),
    include_package_data=False,
    install_requires=[
        "numpy", "tqdm", "Pillow"
    ],
    project_urls={
        'Documentation': 'https://nn-analysis.github.io/nn_tuning/nn_tuning.html',
    },
    python_requires=">=3.6",
)
