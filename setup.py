#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy>=1.18",
    "opencv-python>=4.1.2",
    "scipy>=1.6.0",
    "tqdm>=4.45.0",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]

setup(
    author="RadioactiveDroid",
    author_email="radioactivedroid@gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    description="This package allows for reading a BlockParty floor as a 48x48 pixel image and provides analytics including optimized standing positions.",
    entry_points={"console_scripts": ["bpoptimizer=bpoptimizer.cli:main"]},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="bpoptimizer",
    name="bpoptimizer",
    packages=find_packages(include=["bpoptimizer", "bpoptimizer.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/RadioactiveDroid/bpoptimizer",
    version="2.1.0",
    zip_safe=False,
)
