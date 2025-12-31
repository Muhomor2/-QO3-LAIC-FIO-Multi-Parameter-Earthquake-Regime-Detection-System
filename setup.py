#!/usr/bin/env python3
"""
QO3-LAIC-FIO: Multi-Parameter Earthquake Regime Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qo3-laic-fio",
    version="2.0.0",
    author="Igor Chechelnitsky",
    author_email="",
    description="Multi-Parameter Earthquake Regime Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ichechelnitsky/qo3-laic-fio",
    project_urls={
        "Bug Tracker": "https://github.com/ichechelnitsky/qo3-laic-fio/issues",
        "Documentation": "https://github.com/ichechelnitsky/qo3-laic-fio#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "qo3-fio=qo3_fio_ultimate:main",
        ],
    },
)
